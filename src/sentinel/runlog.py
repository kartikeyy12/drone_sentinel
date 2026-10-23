"""Run directory and incremental writers: runs/<UTC timestamp>/{events.jsonl, events.csv, windows.csv, run.json}.

Events are appended and flushed the moment they close, so a crash or a kill loses at most the open event.
"""
from __future__ import annotations

import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

EVENT_FIELDS = ["sensor_id", "event_id", "label", "t_start", "t_end", "duration_s", "peak_prob", "mean_prob", "top_alt_class",
                "snr_est_db", "model_version"]


def make_run_dir(runs_dir: Path, now: datetime | None = None) -> Path:
    """Create runs/<YYYYmmdd-HHMMSS> (UTC); a numeric suffix is added if it already exists."""
    stamp = (now or datetime.now(timezone.utc)).strftime("%Y%m%d-%H%M%S")
    d = runs_dir / stamp
    n = 1
    while d.exists():
        d = runs_dir / f"{stamp}-{n}"
        n += 1
    d.mkdir(parents=True)
    return d


class RunLog:
    def __init__(self, runs_dir: Path, targets: list[str], meta: dict[str, Any], now: datetime | None = None) -> None:
        self.dir = make_run_dir(Path(runs_dir), now)
        self.targets = list(targets)
        self.meta = dict(meta)
        self._jsonl = open(self.dir / "events.jsonl", "a", encoding="utf-8")
        self._events_f = open(self.dir / "events.csv", "w", newline="", encoding="utf-8")
        self._events = csv.DictWriter(self._events_f, fieldnames=EVENT_FIELDS)
        self._events.writeheader()
        self._events_f.flush()
        self._windows_f = open(self.dir / "windows.csv", "w", newline="", encoding="utf-8")
        self._windows = csv.writer(self._windows_f)
        self._windows.writerow(["index", "t_start", "t_end", "label", "gated", "active", "rms_mean", "snr_db"]
                               + [f"p_{t}" for t in self.targets] + [f"smoothed_{t}" for t in self.targets])
        self.n_events = 0
        self.n_windows = 0
        self._write_meta("running")

    def write_event(self, rec: dict[str, Any]) -> None:
        self._jsonl.write(json.dumps(rec) + "\n")
        self._jsonl.flush()
        self._events.writerow({k: ("" if rec.get(k) is None else rec.get(k)) for k in EVENT_FIELDS})
        self._events_f.flush()
        self.n_events += 1

    def write_window(self, r) -> None:
        self._windows.writerow([r.index, f"{r.t_start:.3f}", f"{r.t_end:.3f}", r.label, int(r.gated), "+".join(r.active),
                                f"{r.rms_mean:.6g}", "" if r.snr_db is None else f"{r.snr_db:.2f}"]
                               + [f"{r.p[t]:.5f}" for t in self.targets] + [f"{r.smoothed[t]:.5f}" for t in self.targets])
        self.n_windows += 1
        if self.n_windows % 20 == 0:
            self._windows_f.flush()

    def _write_meta(self, status: str, extra: dict[str, Any] | None = None) -> None:
        d = {**self.meta, "status": status, "n_events": self.n_events, "n_windows": self.n_windows, **(extra or {})}
        tmp = self.dir / "run.json.tmp"
        tmp.write_text(json.dumps(d, indent=2, default=str), encoding="utf-8")
        os.replace(tmp, self.dir / "run.json")

    def close(self, status: str, extra: dict[str, Any] | None = None) -> None:
        for f in (self._jsonl, self._events_f, self._windows_f):
            try:
                f.flush()
                f.close()
            except ValueError:  # already closed
                pass
        self._write_meta(status, {"finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), **(extra or {})})
