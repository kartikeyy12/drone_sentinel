"""Live console monitor.

    python -m sentinel.monitor --source file --file clip.wav [--realtime]
    python -m sentinel.monitor --source tcp            # then run tools/win_mic_stream.py in Windows
    python -m sentinel.monitor --source mic            # PortAudio input device

Shows the current class, window probability against its threshold, estimated SNR and a scrolling list of closed events.
Every closed event is appended to runs/<timestamp>/events.jsonl and events.csv at once. Ctrl+C stops cleanly: the open event
is closed and written.
"""
from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from . import alerting as al
from .bundle import SentinelModel, load_model
from .config import load_config, project_path, setup_logging
from .engine import StreamEngine, WindowResult
from .runlog import RunLog
from .sources import AudioSource, SourceUnavailable, open_source

log = logging.getLogger("monitor")


@dataclass
class MonitorState:
    source: str
    sensor_id: str
    targets: list[str]
    thresholds: dict[str, float]
    last: WindowResult | None = None
    events: deque = field(default_factory=lambda: deque(maxlen=200))
    n_events: int = 0
    n_windows: int = 0
    stream_time: float = 0.0
    dropped_s: float = 0.0
    waiting: str | None = None           # message shown until the first audio arrives
    status: str = "running"


def prob_bar(p: float, width: int = 30, marker: float | None = None) -> str:
    """Text bar for a probability in [0, 1]; `marker` puts a | at the alert threshold."""
    p = min(max(p, 0.0), 1.0)
    n = int(round(p * width))
    cells = ["█"] * n + ["░"] * (width - n)
    if marker is not None:
        cells[min(width - 1, int(marker * width))] = "|"
    return "".join(cells)


def clock(iso: str) -> str:
    """HH:MM:SS (UTC) part of an event timestamp."""
    return iso[11:19]


def render_dashboard(st: MonitorState, shown: int = 8, bar_width: int = 30) -> Group:
    r = st.last
    head = Text()
    if r is None:
        head.append(st.waiting or "waiting for audio ...", style="yellow")
    elif r.active:
        head.append("▲ ALERT  ", style="bold red")
        head.append(" + ".join(r.active).upper(), style="bold red")
    else:
        head.append("● clear  ", style="green")
        head.append(r.label, style="green" if r.label != al.UNKNOWN else "yellow")
    probs = Table.grid(padding=(0, 2))
    for t in st.targets:
        if r is None:
            probs.add_row(t, "", "", "")
            continue
        probs.add_row(t, prob_bar(r.p[t], bar_width, st.thresholds[t]), f"p {r.p[t]:.2f}",
                      f"smoothed {r.smoothed[t]:.2f}   threshold {st.thresholds[t]:.2f}")
    info = Text()
    if r is not None:
        info.append("SNR estimate  ")
        info.append("n/a (no noise floor yet)" if r.snr_db is None else f"{r.snr_db:+.1f} dB")
        info.append(f"     window {r.index}  t = {st.stream_time:.1f} s")
        if r.gated:
            info.append("     near-silent window, not scored", style="yellow")
    if st.dropped_s > 0:
        info.append(f"\nDROPPED AUDIO {st.dropped_s:.1f} s (processing fell behind)", style="bold yellow")
    ev = Table(title=f"closed events ({st.n_events})", title_justify="left", expand=False)
    for c, j in (("id", "right"), ("class", "left"), ("start UTC", "left"), ("end UTC", "left"), ("dur s", "right"),
                 ("peak", "right"), ("mean", "right"), ("SNR dB", "right")):
        ev.add_column(c, justify=j)
    for e in list(st.events)[-shown:]:
        ev.add_row(str(e["event_id"]), e["label"], clock(e["t_start"]), clock(e["t_end"]), f"{e['duration_s']:.1f}",
                   f"{e['peak_prob']:.2f}", f"{e['mean_prob']:.2f}", "n/a" if e["snr_est_db"] is None else f"{e['snr_est_db']:+.1f}")
    title = f"{st.sensor_id}   source {st.source}   {st.status}"
    return Group(Panel(Group(head, Text(""), probs, Text(""), info), title=title, title_align="left"), ev)


def event_line(rec: dict[str, Any]) -> str:
    snr = "n/a" if rec["snr_est_db"] is None else f"{rec['snr_est_db']:+.1f} dB"
    return (f"EVENT #{rec['event_id']} {rec['label']} {rec['t_start']} -> {rec['t_end']} ({rec['duration_s']:.1f} s) "
            f"peak {rec['peak_prob']:.2f} mean {rec['mean_prob']:.2f} snr {snr}")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m sentinel.monitor", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["mic", "tcp", "file"], required=True)
    ap.add_argument("--file", default=None, help="audio file for --source file")
    ap.add_argument("--realtime", action="store_true", help="with --source file: replay at the speed of the recording")
    ap.add_argument("--zmq", action="store_true", help="publish each closed event over ZeroMQ (PUB, topic and endpoint from m5.zmq)")
    ap.add_argument("--zmq-endpoint", default=None, help="override m5.zmq.endpoint, e.g. tcp://127.0.0.1:5556")
    ap.add_argument("--config", default=None)
    ap.add_argument("--model", default=None, help="model bundle (default: models/pipeline.joblib)")
    ap.add_argument("--runs-dir", default=None)
    ap.add_argument("--t-on", type=float, default=None, help="alert ON threshold on the smoothed probability (default: model threshold)")
    ap.add_argument("--t-off", type=float, default=None)
    ap.add_argument("--device", default=None, help="input device index or name for --source mic")
    ap.add_argument("--tcp-host", default=None)
    ap.add_argument("--tcp-port", type=int, default=None)
    ap.add_argument("--t0", default=None, help="UTC start time of the stream as ISO 8601 (default: now)")
    ap.add_argument("--no-ui", action="store_true", help="plain lines instead of the live display (automatic when not a terminal)")
    return ap


def _parse_t0(s: str | None) -> float:
    """Epoch seconds of an ISO 8601 time (naive times are UTC); now if None."""
    if s is None:
        return time.time()
    d = datetime.fromisoformat(s.replace("Z", "+00:00"))
    return (d.replace(tzinfo=timezone.utc) if d.tzinfo is None else d).timestamp()


def _raise_interrupt(signum, frame) -> None:
    raise KeyboardInterrupt


def run(source: AudioSource, model: SentinelModel, cfg: dict[str, Any], *, runs_dir: Path, ui: bool = True,
        params: dict[str, al.AlertParams] | None = None, t0_wall: float | None = None, console: Console | None = None,
        sensor_id: str | None = None, clock_fn=time.monotonic, publisher: Any = None) -> dict[str, Any]:
    """Drive a source through the engine until it ends or Ctrl+C. Returns a summary dict. Used by main() and by the tests.

    `publisher` (an object with publish(record), for example zmqpub.EventPublisher) gets every closed event right after it is written.
    The caller owns the publisher and closes it.
    """
    console = console or Console()
    eng = StreamEngine(model, cfg, params=params, t0_wall=t0_wall, sensor_id=sensor_id)
    mcfg = cfg["m5"]["monitor"]
    src_path = getattr(source, "path", None)
    meta = {"source": source.describe(), "source_path": str(Path(src_path).resolve()) if src_path else None, "sensor_id": eng.sensor_id, "model_version": model.config_hash,
            "pipeline_hash": model.pipeline_hash, "targets": model.targets, "thresholds": model.thresholds,
            "alert_params": {t: vars(p) for t, p in eng.params.items()},
            "started_at": datetime.fromtimestamp(eng.t0_wall, tz=timezone.utc).isoformat(timespec="milliseconds"),
            "silence_gate_rms": model.silence_gate_rms}
    rl = RunLog(runs_dir, model.targets, meta)
    st = MonitorState(source.describe(), eng.sensor_id, model.targets, model.thresholds,
                      waiting=f"waiting for audio from {source.describe()} ...")
    status = "finished"

    def handle_events(recs: list[dict[str, Any]]) -> None:
        for rec in recs:
            rl.write_event(rec)
            if publisher is not None:
                publisher.publish(rec)
            st.events.append(rec)
            st.n_events += 1
            if not ui:
                console.print(event_line(rec), highlight=False)

    live = Live(render_dashboard(st, mcfg["events_shown"]), console=console, refresh_per_second=mcfg["refresh_hz"],
                transient=False) if ui else None
    last_draw = 0.0
    try:
        if live:
            live.start()
        for chunk in source.chunks():
            wins, recs = eng.push(chunk)
            for w in wins:
                rl.write_window(w)
            st.n_windows, st.stream_time = eng.n_windows, eng.stream_time
            st.dropped_s = source.dropped_samples / source.sr
            if eng.last is not None:
                st.last = eng.last
            handle_events(recs)
            if live and clock_fn() - last_draw >= 1.0 / mcfg["refresh_hz"]:
                live.update(render_dashboard(st, mcfg["events_shown"]))
                last_draw = clock_fn()
    except KeyboardInterrupt:
        status = "interrupted"
    except Exception:
        status = "failed"
        raise
    finally:
        handle_events(eng.flush())
        st.status = status
        if live:
            live.update(render_dashboard(st, mcfg["events_shown"]))
            live.stop()
        source.close()
        summary = {"status": status, "n_windows": eng.n_windows, "n_gated_windows": eng.n_gated, "n_events": rl.n_events,
                   "n_dropped_short_events": eng.n_dropped_short, "stream_seconds": round(eng.stream_time, 3),
                   "dropped_audio_s": round(source.dropped_samples / source.sr, 3),
                   "zmq": None if publisher is None else {"endpoint": getattr(publisher, "endpoint", None), "sent": publisher.n_sent,
                                                            "failed": publisher.n_failed},
                   "median_window_ms": round(float(sorted(eng.compute_ms)[len(eng.compute_ms) // 2]), 2) if eng.compute_ms else None}
        rl.close(status, summary)
    summary["run_dir"] = str(rl.dir)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(logging.WARNING)
    console = Console()
    cfg = load_config(args.config)
    model_path = Path(args.model) if args.model else project_path(cfg, "models") / "pipeline.joblib"
    try:
        model = load_model(model_path, cfg)
    except FileNotFoundError:
        console.print(f"model not found: {model_path}. Build it with: make final")
        return 1
    except ValueError as e:
        console.print(f"{e}")
        return 1
    params = None
    if args.t_on is not None or args.t_off is not None:
        params = {}
        for t in model.targets:
            a = dict(cfg["m5"]["alert"])
            a["t_on"] = args.t_on if args.t_on is not None else a["t_on"]
            a["t_off"] = args.t_off if args.t_off is not None else (None if args.t_on is not None else a["t_off"])
            params[t] = al.AlertParams.from_cfg(a, model.thresholds[t])
    try:
        source = open_source(args.source, cfg, file=args.file, realtime=args.realtime, device=args.device,
                             host=args.tcp_host, port=args.tcp_port)
    except SourceUnavailable as e:
        console.print(f"cannot open the {args.source} source: {e}")
        return 1
    publisher = None
    if args.zmq:
        try:
            from .zmqpub import EventPublisher
            publisher = EventPublisher(args.zmq_endpoint or cfg["m5"]["zmq"]["endpoint"], cfg["m5"]["zmq"]["topic"])
        except ImportError:
            source.close()
            console.print("--zmq needs pyzmq: pip install pyzmq")
            return 1
        except Exception as e:                       # noqa: BLE001 - port in use, bad endpoint
            source.close()
            console.print(f"cannot start the ZeroMQ publisher: {e}")
            return 1
        console.print(f"publishing events on {publisher.endpoint}, topic {publisher.topic.decode()}", highlight=False)
    ui = not args.no_ui and sys.stdout.isatty()
    signal.signal(signal.SIGTERM, _raise_interrupt)
    try:
        summary = run(source, model, cfg, runs_dir=Path(args.runs_dir) if args.runs_dir else project_path(cfg, "runs"), ui=ui,
                      params=params, t0_wall=_parse_t0(args.t0), console=console, publisher=publisher)
    except SourceUnavailable as e:
        console.print(f"source failed: {e}")
        return 1
    finally:
        if publisher is not None:
            publisher.close()
    console.print(f"\n{summary['status']}: {summary['n_events']} events, {summary['n_windows']} windows "
                  f"({summary['stream_seconds']:.1f} s of audio), run directory {summary['run_dir']}", highlight=False)
    if summary.get("zmq"):
        console.print(f"ZeroMQ: {summary['zmq']['sent']} events published, {summary['zmq']['failed']} failed", highlight=False)
    if summary["dropped_audio_s"] > 0:
        console.print(f"warning: {summary['dropped_audio_s']:.1f} s of audio was dropped because processing fell behind", style="yellow")
    return 0


if __name__ == "__main__":
    sys.exit(main())
