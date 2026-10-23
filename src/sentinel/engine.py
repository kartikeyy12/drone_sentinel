"""Streaming engine: audio chunks in, window probabilities, alert state and closed events out.

Chain per window (same as training, with the one causal difference):
  chunk -> causal band-pass (sosfilt, state carried, starts from zero state, no DC removal) -> 1 s window every 0.5 s
        -> 146 features -> z-score, PCA, SVM -> Platt probability -> silence gate -> alerting.step per target class.

Differences from the offline chain that built the training features: the band-pass is causal instead of zero-phase
(scripts/check_streaming.py measures what that costs) and there is no clip-level DC removal (the band-pass low cut removes
DC). Only the `none` variant is supported; the denoisers work on whole clips.

The engine has no clock and no I/O. Times are stream seconds (sample count / sr); `t0_wall` (epoch seconds) maps them to
UTC for the event records.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np

from . import alerting as al
from .bundle import SentinelModel
from .features import feature_names, window_features
from .preprocess import StreamingBandpass, bandpass_sos


def utc_iso(epoch_s: float) -> str:
    """ISO 8601 UTC with millisecond resolution, e.g. 2026-10-12T10:15:03.500Z (the hop is 0.5 s)."""
    ms = int(round(epoch_s * 1000))
    return datetime.fromtimestamp(ms // 1000, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S") + f".{ms % 1000:03d}Z"


@dataclass(frozen=True)
class WindowResult:
    index: int
    t_start: float
    t_end: float
    p: dict[str, float]                    # calibrated probability per target class (0 when gated)
    smoothed: dict[str, float]             # EMA of p per target class
    gated: bool
    label: str                             # silence, ambient, unknown or the winning target class
    active: tuple[str, ...]                # classes whose alert is ON after this window
    rms_mean: float
    snr_db: float | None                   # this window's power against the current noise floor
    floor: float | None
    compute_ms: float


class NoiseFloor:
    """Rolling noise floor: a percentile of the frame powers of the recent background."""

    def __init__(self, percentile: float = 10.0, history_frames: int = 3000, min_frames: int = 50) -> None:
        self.percentile, self.history, self.min_frames = float(percentile), int(history_frames), int(min_frames)
        self._p = np.zeros(0)

    def update(self, powers: np.ndarray) -> None:
        if len(powers):
            self._p = np.concatenate([self._p, np.asarray(powers, np.float64)])[-self.history:]

    @property
    def n_frames(self) -> int:
        return len(self._p)

    @property
    def value(self) -> float | None:
        return al.noise_floor(self._p, self.percentile, self.min_frames)


class StreamEngine:
    def __init__(self, model: SentinelModel, cfg: dict[str, Any], params: dict[str, al.AlertParams] | None = None,
                 t0_wall: float | None = None, sensor_id: str | None = None) -> None:
        if model.variant != "none":
            raise NotImplementedError(f"the streaming engine supports the `none` variant only (model has {model.variant!r}); the denoisers "
                                      "work on whole clips")
        model.check_compatible(cfg)
        self.model, self.cfg = model, cfg
        c = model.cfg_used
        self.sr = int(c["audio"]["sr"])
        self.win = int(round(c["audio"]["window_s"] * self.sr))
        self.hop = int(round(c["audio"]["hop_s"] * self.sr))
        self._fcfg = c["features"]
        bp = c["preprocess"]["bandpass"]
        self._bp = StreamingBandpass(bandpass_sos(self.sr, bp["low_hz"], bp["high_hz"], bp["order"]))
        self._rms_col = feature_names(self._fcfg).index("rms_mean")
        m5 = cfg["m5"]
        self.sensor_id = sensor_id or m5["sensor_id"]
        self.reject_below = m5["alert"].get("reject_below")
        self.params = params or {t: al.AlertParams.from_cfg(m5["alert"], model.thresholds[t]) for t in model.targets}
        self.trackers = {t: al.TrackerState(label=t) for t in model.targets}
        sn = m5["snr"]
        self._frame_len = max(1, int(round(sn["frame_s"] * self.sr)))
        self.floor = NoiseFloor(sn["percentile"], int(round(sn["history_s"] / sn["frame_s"])), sn["min_frames"])
        self.t0_wall = time.time() if t0_wall is None else float(t0_wall)
        self._buf = np.zeros(0, np.float32)
        self._buf_start = 0              # absolute sample index of _buf[0]
        self._next = 0                   # absolute start sample of the next window
        self._seen = 0                   # samples received so far
        self.n_windows = 0
        self.n_gated = 0
        self.n_events = 0
        self.compute_ms: list[float] = []
        self.last: WindowResult | None = None

    # ---- properties ------------------------------------------------------------------------------------------------
    @property
    def stream_time(self) -> float:
        return self._seen / self.sr

    @property
    def active(self) -> tuple[str, ...]:
        return tuple(t for t, s in self.trackers.items() if s.active)

    @property
    def n_dropped_short(self) -> int:
        return sum(s.n_dropped_short for s in self.trackers.values())

    # ---- events ----------------------------------------------------------------------------------------------------
    def record(self, ev: al.Event) -> dict[str, Any]:
        """The event as the JSON record of the spec (stream times mapped to UTC)."""
        self.n_events += 1
        return {"sensor_id": self.sensor_id, "event_id": self.n_events, "label": ev.label,
                "t_start": utc_iso(self.t0_wall + ev.t_start), "t_end": utc_iso(self.t0_wall + ev.t_end),
                "duration_s": round(ev.duration_s, 3), "peak_prob": round(ev.peak_prob, 4), "mean_prob": round(ev.mean_prob, 4),
                "top_alt_class": ev.top_alt_class,
                "snr_est_db": None if ev.snr_est_db is None else round(ev.snr_est_db, 1), "model_version": self.model.config_hash}

    # ---- processing ------------------------------------------------------------------------------------------------
    def push(self, chunk: np.ndarray) -> tuple[list[WindowResult], list[dict[str, Any]]]:
        """Feed audio (float in [-1, 1] or int16). Returns the windows completed by it and the events that became final."""
        x = np.asarray(chunk)
        x = x.astype(np.float32) / 32768.0 if x.dtype == np.int16 else x.astype(np.float32, copy=False)
        y = self._bp.process(x)
        self._seen += len(y)
        self._buf = np.concatenate([self._buf, y])
        results: list[WindowResult] = []
        records: list[dict[str, Any]] = []
        while self._next + self.win <= self._buf_start + len(self._buf):
            a = self._next - self._buf_start
            r, evs = self._process_window(self._buf[a:a + self.win], self._next)
            results.append(r)
            records += evs
            self._next += self.hop
        drop = self._next - self._buf_start
        if drop > 0:
            self._buf, self._buf_start = self._buf[drop:], self._next
        return results, records

    def _process_window(self, win: np.ndarray, start: int) -> tuple[WindowResult, list[dict[str, Any]]]:
        t0 = time.perf_counter()
        k = self.n_windows
        feats = window_features(win, self.sr, self._fcfg)
        rms = float(feats[self._rms_col])
        p, gated = self.model.score_features(feats[None, :], np.array([rms]))
        pt = {t: float(p[0, i]) for i, t in enumerate(self.model.targets)}
        g = bool(gated[0])
        new = win if k == 0 else win[self.win - self.hop:]
        power = float(np.mean(al.frame_powers(new, self._frame_len))) if len(new) >= self._frame_len else 0.0
        floor = self.floor.value
        p_amb = max(0.0, 1.0 - max(pt.values(), default=0.0))
        t_start, t_end = start / self.sr, (start + self.win) / self.sr
        records: list[dict[str, Any]] = []
        smoothed: dict[str, float] = {}
        for t, st in self.trackers.items():
            alt = tuple((o, q) for o, q in pt.items() if o != t) + ((al.AMBIENT, p_amb),)
            st, evs = al.step(self.params[t], st, al.Obs(t_start, t_end, pt[t], power, floor, alt))
            self.trackers[t] = st
            smoothed[t] = float(st.smoothed)
            records += [self.record(e) for e in evs]
        label = al.window_label(pt, self.model.thresholds, g, self.reject_below)
        background = not g and not any(pt[t] >= self.model.thresholds[t] for t in pt) and not self.active
        if background or g:
            self.floor.update(al.frame_powers(new, self._frame_len))
        self.n_windows += 1
        self.n_gated += int(g)
        ms = (time.perf_counter() - t0) * 1000.0
        self.compute_ms.append(ms)
        r = WindowResult(k, t_start, t_end, pt, smoothed, g, label, self.active, rms,
                         al.snr_db(float(np.mean(win.astype(np.float64) ** 2)), floor), floor, ms)
        self.last = r
        return r, records

    def flush(self) -> list[dict[str, Any]]:
        """End of stream: close open events (as if the alert went OFF) and release held-back ones."""
        out: list[dict[str, Any]] = []
        for t, st in self.trackers.items():
            st, evs = al.flush(self.params[t], st)
            self.trackers[t] = st
            out += [self.record(e) for e in evs]
        return out


def run_offline(model: SentinelModel, cfg: dict[str, Any], x: np.ndarray, chunk_s: float | None = None, t0_wall: float = 0.0,
                params: dict[str, al.AlertParams] | None = None) -> tuple[list[WindowResult], list[dict[str, Any]], StreamEngine]:
    """Run a whole array through a fresh engine in chunks and flush. Returns (windows, event records, engine)."""
    eng = StreamEngine(model, cfg, params=params, t0_wall=t0_wall)
    n = max(1, int(round((chunk_s or cfg["m5"]["stream"]["chunk_s"]) * eng.sr)))
    wins: list[WindowResult] = []
    recs: list[dict[str, Any]] = []
    for i in range(0, len(x), n):
        w, r = eng.push(x[i:i + n])
        wins += w
        recs += r
    recs += eng.flush()
    return wins, recs, eng
