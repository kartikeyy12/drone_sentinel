"""Alert logic: turn per-window probabilities into events. Pure functions and an immutable state, no audio, no clock.

For one tracked class (one state machine per class):
  1. EMA          s_t = a * p_t + (1 - a) * s_(t-1), s_0 = p_0.
  2. Onset        the alert turns ON when the smoothed s reaches t_on in at least N of the last M windows. The event is
                  back-dated to the first of those windows, so its start is where the evidence started, not where the
                  N-th window arrived. `Event.t_alert` keeps the moment the alert was raised (end of that window).
  3. Hysteresis   the alert turns OFF when s stays below t_off (t_off < t_on) for K consecutive windows. The event ends at
                  the end of the last window that was not below t_off.
  4. Duration     a closed event shorter than `min_event_s` is dropped.
  5. Gap merging  an event that starts within `merge_gap_s` of the previous event's end is merged into it (the windows in
                  between are included in the statistics). The previous event is held back until that can no longer happen.
  6. SNR          power-subtraction estimate against a rolling noise floor (10th percentile of frame power), see `snr_db`.

Times are seconds on the stream clock (window k starts at k * hop). Probabilities are the calibrated window probabilities
of the model, so `peak_prob` and `mean_prob` are on that scale.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

AMBIENT = "ambient"
UNKNOWN = "unknown"
SILENCE = "silence"
_EPS = 1e-9


# ---- parameters ------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class AlertParams:
    t_on: float
    t_off: float
    ema_alpha: float = 0.5
    n_on: int = 3
    m_window: int = 5
    k_off: int = 4
    min_event_s: float = 2.0
    merge_gap_s: float = 3.0

    def __post_init__(self) -> None:
        if not 0 < self.t_off < self.t_on <= 1:
            raise ValueError(f"need 0 < t_off < t_on <= 1, got t_off={self.t_off}, t_on={self.t_on}")
        if not 0 < self.ema_alpha <= 1:
            raise ValueError("ema_alpha must be in (0, 1]")
        if not 1 <= self.n_on <= self.m_window:
            raise ValueError(f"need 1 <= n_on <= m_window, got n_on={self.n_on}, m_window={self.m_window}")
        if self.k_off < 1:
            raise ValueError("k_off must be at least 1")
        if self.min_event_s < 0 or self.merge_gap_s < 0:
            raise ValueError("min_event_s and merge_gap_s must be >= 0")

    @classmethod
    def from_cfg(cls, alert_cfg: Mapping[str, Any], default_t_on: float) -> "AlertParams":
        """Build from the `m5.alert` config block. A null t_on means `default_t_on` (the model's saved threshold); a null
        t_off means t_off_ratio * t_on."""
        t_on = float(alert_cfg["t_on"]) if alert_cfg.get("t_on") is not None else float(default_t_on)
        t_off = float(alert_cfg["t_off"]) if alert_cfg.get("t_off") is not None else float(alert_cfg["t_off_ratio"]) * t_on
        return cls(t_on=t_on, t_off=t_off, ema_alpha=float(alert_cfg["ema_alpha"]), n_on=int(alert_cfg["n_on"]),
                   m_window=int(alert_cfg["m_window"]), k_off=int(alert_cfg["k_off"]),
                   min_event_s=float(alert_cfg["min_event_s"]), merge_gap_s=float(alert_cfg["merge_gap_s"]))


# ---- small pure functions --------------------------------------------------------------------------------------------

def ema_update(prev: float | None, x: float, alpha: float) -> float:
    """One EMA step. The first value passes through unchanged."""
    return float(x) if prev is None else float(alpha * x + (1.0 - alpha) * prev)


def frame_powers(x: np.ndarray, frame_len: int) -> np.ndarray:
    """Mean power of consecutive non-overlapping frames; a trailing partial frame is dropped."""
    n = len(x) // frame_len
    if n == 0:
        return np.zeros(0)
    return np.mean(np.asarray(x[: n * frame_len], np.float64).reshape(n, frame_len) ** 2, axis=1)


def noise_floor(powers: np.ndarray, percentile: float = 10.0, min_frames: int = 1) -> float | None:
    """Noise floor power: the given percentile of frame power. None if there are fewer than `min_frames` frames."""
    if len(powers) < max(1, min_frames):
        return None
    return float(np.percentile(powers, percentile))


def snr_db(power: float, floor: float | None, min_floor: float = 1e-10, lowest_db: float = -20.0) -> float | None:
    """Power-subtraction SNR estimate: 10 log10((P - N) / N) for signal-plus-noise power P and noise floor N.

    Values below `lowest_db` (including P <= N) are reported as `lowest_db`. None when there is no usable floor (no
    estimate yet, or digital silence below `min_floor`), since the ratio is then meaningless.
    """
    if floor is None or not math.isfinite(floor) or floor < min_floor or not math.isfinite(power):
        return None
    ratio = power / floor - 1.0
    return float(max(10.0 * math.log10(max(ratio, 1e-12)), lowest_db))


def window_label(p_targets: Mapping[str, float], thresholds: Mapping[str, float], gated: bool = False,
                 reject_below: float | None = None) -> str:
    """Label of one window for display and window-level statistics (not an input to the event logic).

    silence if gated; otherwise the alerting target class with the highest probability among those at or above their
    threshold; otherwise ambient. With `reject_below`, a window whose largest class probability (ambient counts as
    1 - max target probability) is below it is labelled unknown instead.
    """
    if gated:
        return SILENCE
    hit = {c: p for c, p in p_targets.items() if p >= thresholds[c]}
    if reject_below is not None:
        top = max([*p_targets.values(), 1.0 - max(p_targets.values(), default=0.0)])
        if top < reject_below:
            return UNKNOWN
    return max(hit, key=hit.get) if hit else AMBIENT


# ---- observations, events and state -----------------------------------------------------------------------------------

@dataclass(frozen=True)
class Obs:
    """What the tracker needs to know about one window."""
    t_start: float
    t_end: float
    p: float                                   # calibrated probability of the tracked class (0 for a gated window)
    power: float = 0.0                         # in-band mean power of the audio this window adds
    floor: float | None = None                 # noise floor power at this window, None if there is no estimate
    alt: tuple[tuple[str, float], ...] = ()    # (class, probability) of the other classes, ambient included


@dataclass(frozen=True)
class Event:
    label: str
    t_start: float
    t_end: float
    t_alert: float
    peak_prob: float
    mean_prob: float
    top_alt_class: str
    snr_est_db: float | None
    n_windows: int

    @property
    def duration_s(self) -> float:
        return self.t_end - self.t_start


@dataclass(frozen=True)
class _Run:
    windows: tuple[Obs, ...]
    t_alert: float

    @property
    def t_start(self) -> float:
        return self.windows[0].t_start

    @property
    def t_end(self) -> float:
        return self.windows[-1].t_end


@dataclass(frozen=True)
class TrackerState:
    label: str
    ema: float | None = None
    hist: tuple[tuple[Obs, float], ...] = ()      # last M windows with their smoothed probability
    below_run: int = 0                            # consecutive windows below t_off while an event is open
    open: _Run | None = None                      # the open event; its tail may hold the below-t_off windows
    pending: _Run | None = None                   # closed event held back for possible merging
    after_pending: tuple[Obs, ...] = ()           # windows seen since `pending` closed
    n_dropped_short: int = 0

    @property
    def active(self) -> bool:
        """True while an alert is ON."""
        return self.open is not None

    @property
    def smoothed(self) -> float | None:
        return self.ema


def event_snr(windows: Sequence[Obs]) -> float | None:
    """SNR of an event: its mean window power against the median noise floor over its windows."""
    floors = [w.floor for w in windows if w.floor is not None]
    if not floors:
        return None
    return snr_db(float(np.mean([w.power for w in windows])), float(np.median(floors)))


def make_event(label: str, run: _Run) -> Event:
    p = np.array([w.p for w in run.windows])
    acc: dict[str, list[float]] = {}
    for w in run.windows:
        for c, q in w.alt:
            acc.setdefault(c, []).append(q)
    alt = max(acc, key=lambda c: float(np.mean(acc[c]))) if acc else AMBIENT
    return Event(label=label, t_start=run.t_start, t_end=run.t_end, t_alert=run.t_alert, peak_prob=float(p.max()),
                 mean_prob=float(p.mean()), top_alt_class=alt, snr_est_db=event_snr(run.windows), n_windows=len(run.windows))


def _long_enough(run: _Run, params: AlertParams) -> bool:
    return run.t_end - run.t_start >= params.min_event_s - _EPS


# ---- the state machine --------------------------------------------------------------------------------------------------

def step(params: AlertParams, st: TrackerState, obs: Obs) -> tuple[TrackerState, list[Event]]:
    """Advance the tracker by one window. Returns the new state and the events that became final at this window."""
    s = ema_update(st.ema, obs.p, params.ema_alpha)
    hist = (st.hist + ((obs, s),))[-params.m_window:]
    events: list[Event] = []
    open_, pending, after, below, dropped = st.open, st.pending, st.after_pending, st.below_run, st.n_dropped_short

    if open_ is None:
        if pending is not None:
            after = after + (obs,)
        flags = [sm >= params.t_on for _, sm in hist]
        if sum(flags) >= params.n_on:
            first = flags.index(True)
            run = tuple(o for o, _ in hist[first:])
            if pending is not None and run[0].t_start - pending.t_end <= params.merge_gap_s + _EPS:
                # continues the previous event: its windows, the windows in between, then the new evidence
                between = tuple(o for o in after if pending.windows[-1].t_start < o.t_start < run[0].t_start - _EPS)
                open_ = _Run(pending.windows + between + run, t_alert=pending.t_alert)
            else:
                if pending is not None:
                    events.append(make_event(st.label, pending))
                open_ = _Run(run, t_alert=obs.t_end)
            pending, after, below = None, (), 0
    else:
        open_ = replace(open_, windows=open_.windows + (obs,))
        below = 0 if s >= params.t_off else below + 1
        if below >= params.k_off:
            run = _Run(open_.windows[:-below], open_.t_alert)
            tail = open_.windows[-below:]
            open_, below, hist = None, 0, ()          # evidence from this event must not re-trigger the next one
            if not _long_enough(run, params):
                dropped += 1
            elif params.merge_gap_s > 0:
                pending, after = run, tail
            else:
                events.append(make_event(st.label, run))

    if open_ is None and pending is not None:        # can a future event still merge into it?
        earliest = min([o.t_start for o, sm in hist if sm >= params.t_on] + [obs.t_start])
        if earliest - pending.t_end > params.merge_gap_s + _EPS:
            events.append(make_event(st.label, pending))
            pending, after = None, ()
    return replace(st, ema=s, hist=hist, below_run=below, open=open_, pending=pending, after_pending=after,
                   n_dropped_short=dropped), events


def flush(params: AlertParams, st: TrackerState) -> tuple[TrackerState, list[Event]]:
    """End of stream: close the open event as if the alert had gone OFF, and release any held-back event."""
    events: list[Event] = []
    if st.open is not None:
        windows = st.open.windows[: len(st.open.windows) - st.below_run] if st.below_run else st.open.windows
        run = _Run(windows, st.open.t_alert)
        if _long_enough(run, params):
            events.append(make_event(st.label, run))
        else:
            st = replace(st, n_dropped_short=st.n_dropped_short + 1)
    elif st.pending is not None:
        events.append(make_event(st.label, st.pending))
    return replace(st, open=None, pending=None, after_pending=(), hist=(), below_run=0), events


def detect_events(params: AlertParams, observations: Iterable[Obs], label: str) -> list[Event]:
    """Run the state machine over a whole series of windows (the batch form used for scenario and false-alarm runs)."""
    st, out = TrackerState(label=label), []
    for o in observations:
        st, ev = step(params, st, o)
        out += ev
    return out + flush(params, st)[1]
