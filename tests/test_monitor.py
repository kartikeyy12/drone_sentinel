"""Run directory, event files, dashboard rendering and the monitor loop (clean end, Ctrl+C, errors, command line)."""
from __future__ import annotations

import csv
import io
import json
import re
from datetime import datetime, timezone

import numpy as np
import pytest
import soundfile as sf
from rich.console import Console

from helpers_stream import SR, noise, scene, tone_stack
from sentinel import monitor as mon
from sentinel.bundle import save_model
from sentinel.engine import WindowResult
from sentinel.runlog import EVENT_FIELDS, RunLog, make_run_dir
from sentinel.sources import AudioSource, FileSource, SourceUnavailable

REC = {"sensor_id": "acoustic-01", "event_id": 1, "label": "aircraft", "t_start": "2026-10-12T10:15:03.000Z",
       "t_end": "2026-10-12T10:15:41.000Z", "duration_s": 38.0, "peak_prob": 0.97, "mean_prob": 0.91,
       "top_alt_class": "ambient", "snr_est_db": 9.4, "model_version": "abc"}


def wres(i=3, p=0.6, active=("aircraft",), label="aircraft", gated=False, snr=8.0):
    return WindowResult(i, i * 0.5, i * 0.5 + 1, {"aircraft": p}, {"aircraft": p}, gated, label, active, 0.05, snr, 1e-4, 5.0)


def text(renderable, width=100):
    c = Console(width=width, record=True, force_terminal=False, file=io.StringIO())
    c.print(renderable)
    return c.export_text()


# ---- run directory ---------------------------------------------------------------------------------------------------------

def test_run_dirs_are_utc_timestamps_and_never_collide(tmp_path):
    now = datetime(2026, 10, 12, 10, 15, 3, tzinfo=timezone.utc)
    a, b = make_run_dir(tmp_path, now), make_run_dir(tmp_path, now)
    assert a.name == "20261012-101503" and b.name == "20261012-101503-1" and a.is_dir() and b.is_dir()


def test_events_are_on_disk_as_soon_as_they_are_written(tmp_path):
    rl = RunLog(tmp_path, ["aircraft"], {"x": 1})
    rl.write_event(REC)
    # read while the log is still open: nothing waits for close()
    assert json.loads((rl.dir / "events.jsonl").read_text().splitlines()[0]) == REC
    rows = list(csv.DictReader((rl.dir / "events.csv").open()))
    assert list(rows[0]) == EVENT_FIELDS and rows[0]["event_id"] == "1" and rows[0]["snr_est_db"] == "9.4"
    assert json.loads((rl.dir / "run.json").read_text())["status"] == "running"
    rl.write_event({**REC, "event_id": 2, "snr_est_db": None})
    rl.close("finished", {"extra": 5})
    rows = list(csv.DictReader((rl.dir / "events.csv").open()))
    assert rows[1]["snr_est_db"] == "" and len((rl.dir / "events.jsonl").read_text().splitlines()) == 2
    meta = json.loads((rl.dir / "run.json").read_text())
    assert meta["status"] == "finished" and meta["n_events"] == 2 and meta["extra"] == 5 and meta["x"] == 1
    rl.close("finished")                         # closing twice is harmless


def test_window_rows_have_one_probability_column_per_target(tmp_path):
    rl = RunLog(tmp_path, ["aircraft"], {})
    rl.write_window(wres(0, 0.25, active=(), label="ambient", snr=None))
    rl.write_window(wres(1, 0.75))
    rl.close("finished")
    rows = list(csv.DictReader((rl.dir / "windows.csv").open()))
    assert [r["label"] for r in rows] == ["ambient", "aircraft"] and rows[0]["snr_db"] == "" and rows[1]["p_aircraft"] == "0.75000"
    assert rows[1]["active"] == "aircraft" and rows[0]["smoothed_aircraft"] == "0.25000"


# ---- dashboard ------------------------------------------------------------------------------------------------------------

def test_prob_bar_fill_and_threshold_marker():
    assert mon.prob_bar(0.0, 10) == "░" * 10 and mon.prob_bar(1.0, 10) == "█" * 10
    assert mon.prob_bar(0.5, 10) == "█" * 5 + "░" * 5
    b = mon.prob_bar(0.2, 10, marker=0.6)
    assert len(b) == 10 and b[6] == "|" and b.startswith("██")
    assert mon.prob_bar(7.0, 4) == "████" and mon.prob_bar(-1, 4) == "░░░░"


def test_dashboard_states():
    st = mon.MonitorState("file:x.wav", "acoustic-01", ["aircraft"], {"aircraft": 0.3}, waiting="waiting for tcp client")
    out = text(mon.render_dashboard(st))
    assert "waiting for tcp client" in out and "closed events (0)" in out and "acoustic-01" in out
    st.last = wres()
    st.events.append(REC)
    st.n_events = 1
    out = text(mon.render_dashboard(st))
    assert "ALERT" in out and "AIRCRAFT" in out and "p 0.60" in out and "+8.0 dB" in out and "threshold 0.30" in out
    assert "10:15:03" in out and "38.0" in out and "+9.4" in out and "closed events (1)" in out
    st.last = wres(active=(), label="ambient", p=0.02, snr=None)
    out = text(mon.render_dashboard(st))
    assert "clear" in out and "ambient" in out and "ALERT" not in out and "n/a" in out
    st.last = wres(active=(), label="silence", gated=True)
    st.dropped_s = 2.5
    out = text(mon.render_dashboard(st))
    assert "not scored" in out and "DROPPED AUDIO 2.5 s" in out


def test_dashboard_lists_only_the_most_recent_events():
    st = mon.MonitorState("s", "id", ["aircraft"], {"aircraft": 0.3})
    for i in range(1, 15):
        st.events.append({**REC, "event_id": i})
    out = text(mon.render_dashboard(st, shown=3))
    assert re.findall(r"│\s*(\d+)\s*│\s*aircraft", out) == ["12", "13", "14"]


def test_event_line_with_and_without_snr():
    assert "snr +9.4 dB" in mon.event_line(REC) and "#1 aircraft" in mon.event_line(REC)
    assert "snr n/a" in mon.event_line({**REC, "snr_est_db": None})


# ---- the loop --------------------------------------------------------------------------------------------------------------

def _src(x, chunk_s=0.25):
    return FileSource("mem.wav", SR, chunk_s, samples=x)


def test_run_writes_events_windows_and_metadata(toy, tmp_path):
    cfg, model = toy
    s = mon.run(_src(scene(8, 10, 8)), model, cfg, runs_dir=tmp_path, ui=False, t0_wall=1_760_000_000.0,
                console=Console(file=io.StringIO()))
    d = tmp_path / s["run_dir"].split("/")[-1]
    ev = [json.loads(l) for l in (d / "events.jsonl").read_text().splitlines()]
    assert s["status"] == "finished" and s["n_events"] == len(ev) == 1 and ev[0]["label"] == "aircraft"
    assert len(list(csv.DictReader((d / "windows.csv").open()))) == s["n_windows"] == 51
    meta = json.loads((d / "run.json").read_text())
    assert meta["status"] == "finished" and meta["source"] == "file:mem.wav" and meta["model_version"] == model.config_hash
    assert meta["alert_params"]["aircraft"]["n_on"] == 3 and meta["dropped_audio_s"] == 0 and meta["median_window_ms"] > 0
    assert meta["stream_seconds"] == pytest.approx(26.0)


def test_plain_mode_prints_each_event_as_it_closes(toy, tmp_path):
    cfg, model = toy
    buf = io.StringIO()
    mon.run(_src(scene(8, 10, 8)), model, cfg, runs_dir=tmp_path, ui=False, console=Console(file=buf, width=200))
    assert buf.getvalue().count("EVENT #1 aircraft") == 1


def test_live_ui_renders_and_ends_with_the_final_state(toy, tmp_path):
    cfg, model = toy
    buf = io.StringIO()
    mon.run(_src(scene(8, 10, 8)), model, cfg, runs_dir=tmp_path, ui=True, console=Console(file=buf, width=100, force_terminal=True),
            clock_fn=iter(range(0, 10_000, 1)).__next__)          # a clock that always says "time to redraw"
    out = buf.getvalue()
    assert "closed events (1)" in out and "finished" in out


class Interrupted(AudioSource):
    """Delivers some audio, then behaves as if the user pressed Ctrl+C."""
    sr, chunk_samples, name = SR, 4000, "test:interrupt"

    def __init__(self, x):
        self.x, self.closed = x, False

    def chunks(self):
        for i in range(0, len(self.x), 4000):
            yield self.x[i:i + 4000]
        raise KeyboardInterrupt

    def close(self):
        self.closed = True


def test_ctrl_c_closes_the_open_event_and_writes_it(toy, tmp_path):
    cfg, model = toy
    x = np.concatenate([noise(8, seed=1), tone_stack(10, seed=2) + noise(10, seed=3)])      # the tone is still on at the end
    s = Interrupted(x)
    out = mon.run(s, model, cfg, runs_dir=tmp_path, ui=False, console=Console(file=io.StringIO()))
    assert out["status"] == "interrupted" and out["n_events"] == 1 and s.closed
    d = next(tmp_path.iterdir())
    ev = json.loads((d / "events.jsonl").read_text().splitlines()[0])
    assert ev["label"] == "aircraft" and ev["duration_s"] >= 8
    assert json.loads((d / "run.json").read_text())["status"] == "interrupted"


class Failing(AudioSource):
    sr, chunk_samples, name = SR, 4000, "test:fail"

    def chunks(self):
        yield np.zeros(4000, np.float32)
        raise SourceUnavailable("device unplugged")


def test_a_failing_source_is_recorded_and_the_error_propagates(toy, tmp_path):
    cfg, model = toy
    with pytest.raises(SourceUnavailable, match="unplugged"):
        mon.run(Failing(), model, cfg, runs_dir=tmp_path, ui=False, console=Console(file=io.StringIO()))
    assert json.loads((next(tmp_path.iterdir()) / "run.json").read_text())["status"] == "failed"


# ---- command line ----------------------------------------------------------------------------------------------------------

@pytest.fixture
def saved(toy, tmp_path):
    cfg, model = toy
    p = tmp_path / "models" / "pipeline.joblib"
    save_model(model, p)
    sf.write(tmp_path / "scene.wav", scene(8, 10, 8), SR)
    return p, tmp_path / "scene.wav"


def test_main_file_source_end_to_end(saved, tmp_path, capsys):
    model_path, wav = saved
    runs = tmp_path / "runs"
    rc = mon.main(["--source", "file", "--file", str(wav), "--model", str(model_path), "--runs-dir", str(runs), "--no-ui",
                   "--t0", "2026-10-12T10:00:00Z"])
    out = capsys.readouterr().out
    assert rc == 0 and "EVENT #1 aircraft" in out and "finished: 1 events" in out
    d = next(runs.iterdir())
    ev = json.loads((d / "events.jsonl").read_text().splitlines()[0])
    assert ev["t_start"].startswith("2026-10-12T10:00:")


def test_main_threshold_override_changes_the_alerting(saved, tmp_path):
    model_path, wav = saved
    runs = tmp_path / "runs"
    assert mon.main(["--source", "file", "--file", str(wav), "--model", str(model_path), "--runs-dir", str(runs), "--no-ui",
                     "--t-on", "0.9999999"]) == 0
    d = next(runs.iterdir())
    assert json.loads((d / "run.json").read_text())["alert_params"]["aircraft"]["t_on"] == pytest.approx(0.9999999)


def test_main_reports_a_missing_model_and_a_missing_file(saved, tmp_path, capsys):
    model_path, wav = saved
    assert mon.main(["--source", "file", "--file", str(wav), "--model", str(tmp_path / "none.joblib"), "--no-ui"]) == 1
    assert "make final" in capsys.readouterr().out
    assert mon.main(["--source", "file", "--file", str(tmp_path / "gone.wav"), "--model", str(model_path), "--no-ui"]) == 1
    assert "not found" in capsys.readouterr().out
    assert mon.main(["--source", "file", "--model", str(model_path), "--no-ui"]) == 1
    assert "--file" in capsys.readouterr().out


def test_main_mic_without_an_input_device_explains_the_alternative(saved, monkeypatch, capsys):
    monkeypatch.setitem(__import__("sys").modules, "sounddevice", None)
    assert mon.main(["--source", "mic", "--model", str(saved[0]), "--no-ui"]) == 1
    assert "tcp" in capsys.readouterr().out
