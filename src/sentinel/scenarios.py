"""M6 building blocks: cross-fitted models, scenario streams with ground truth, event matching, threshold sweep, operating point.

Honesty notes that matter for every number built on this module:
  * The saved model was fitted on all ESC-50 windows. Scoring ESC-50 clips with it would be in-sample. `fold_models`
    therefore returns one model per M3 fold (fitted on the other folds only), and a stream is only ever scored by the model of
    the fold whose recordings it is made from. Calibrators and the silence gate are the saved ones (two parameters fitted on
    out-of-fold scores).
  * "Targets" are ESC-50 aircraft clips. There is no drone class.
  * ESC-50 clips are 5 s long; streams built from them are concatenations with abrupt joins, not continuous recordings.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from . import alerting as al
from .bundle import SentinelModel, final_config
from .dataset import Dataset
from .engine import WindowResult
from .noise import inband_power
from .pipeline import FeaturePipeline


# ---- cross-fitting ---------------------------------------------------------------------------------------------------------

def fold_models(cfg: dict[str, Any], model: SentinelModel, ds: Dataset, folds: list[tuple[np.ndarray, np.ndarray]]) -> list[SentinelModel]:
    """One copy of `model` per fold whose pipeline was fitted on that fold's training windows only."""
    fc = final_config(cfg)
    out = []
    for tr, _ in folds:
        pipe = FeaturePipeline(ds.names, fc["svm"], fc["pca"], fc["drop_level_features"], cfg["m3"]["level_feature_prefixes"],
                               cfg["m3"]["class_weight"], cfg["seed"]).fit(ds.X[tr], ds.y[tr])
        if pipe.classes_ != model.classes:
            raise ValueError(f"fold pipeline classes {pipe.classes_} differ from the model's {model.classes}")
        m = copy.copy(model)
        m.pipeline = pipe
        out.append(m)
    return out


# ---- streams ---------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Segment:
    start_s: float
    end_s: float
    category: str
    hard_negative: bool


def concat_clips(clips: Sequence[tuple[np.ndarray, str, bool]], sr: int, max_s: float | None = None) -> tuple[np.ndarray, list[Segment]]:
    """Join clips end to end (no crossfade). Returns the audio and the segment table. `max_s` truncates the stream."""
    parts, segs, t = [], [], 0.0
    for x, cat, hn in clips:
        if max_s is not None and t >= max_s:
            break
        parts.append(np.asarray(x, np.float32))
        segs.append(Segment(t, t + len(x) / sr, cat, bool(hn)))
        t += len(x) / sr
    x = np.concatenate(parts) if parts else np.zeros(0, np.float32)
    if max_s is not None and len(x) > int(max_s * sr):
        x = x[:int(max_s * sr)]
        segs = [Segment(s.start_s, min(s.end_s, max_s), s.category, s.hard_negative) for s in segs if s.start_s < max_s]
    return x, segs


def place_times(rng: np.random.Generator, n: int, total_s: float, clip_s: float, min_gap_s: float, margin_s: float) -> list[float]:
    """n sorted clip start times, uniformly spread, no overlap, at least min_gap_s between clips, margin_s from both ends."""
    slack = total_s - 2 * margin_s - n * clip_s - (n - 1) * min_gap_s
    if slack < 0:
        raise ValueError(f"{n} clips of {clip_s} s with {min_gap_s} s gaps do not fit in {total_s} s with {margin_s} s margins")
    off = np.sort(rng.uniform(0, slack, n))
    return [float(margin_s + i * (clip_s + min_gap_s) + off[i]) for i in range(n)]


def embed(bg: np.ndarray, targets: Sequence[np.ndarray], starts_s: Sequence[float], snr_db: float, sr: int, cfg: dict[str, Any],
          p_bg: float | None = None) -> np.ndarray:
    """Add each target clip to the background at its start time, scaled so its in-band power is `snr_db` above the in-band
    power of the whole background stream. Mixing happens on the raw audio, before any normalisation."""
    p_bg = inband_power(bg, sr, cfg) if p_bg is None else p_bg
    mix = np.asarray(bg, np.float64).copy()
    for clip, t in zip(targets, starts_s):
        a = int(round(t * sr))
        if a + len(clip) > len(mix):
            raise ValueError("target clip runs past the end of the stream")
        g = np.sqrt(p_bg * 10 ** (snr_db / 10.0) / inband_power(clip, sr, cfg))
        mix[a:a + len(clip)] += g * np.asarray(clip, np.float64)
    return mix.astype(np.float32)


# ---- matching ----------------------------------------------------------------------------------------------------------------

def match_events(truth: Sequence[Mapping[str, float]], events: Sequence[al.Event], tol_s: float = 1.0) -> dict[str, Any]:
    """Match alert events to ground-truth clips. A clip is detected if some event overlaps [t_start - tol, t_end + tol].

    Latency = t_alert of the first matching event minus the clip's true start (can be below the 1 s window length plus
    the N-of-M delay, never below the first window end). Events matching no clip are false alarms.
    """
    used: set[int] = set()
    det, lat = [], []
    for g in truth:
        hit = [i for i, e in enumerate(events) if e.t_start <= g["t_end"] + tol_s and e.t_end >= g["t_start"] - tol_s]
        used.update(hit)
        det.append(bool(hit))
        lat.append(float(events[min(hit)].t_alert - g["t_start"]) if hit else float("nan"))
    return {"detected": det, "latency_s": lat, "false_events": [e for i, e in enumerate(events) if i not in used]}


def obs_from_windows(wins: Sequence[WindowResult], target: str) -> list[al.Obs]:
    """Tracker input from stored engine output. Power is not stored, so SNR is not available from these (sweeps do not need it)."""
    return [al.Obs(w.t_start, w.t_end, w.p[target], 0.0, w.floor, ()) for w in wins]


def poisson_ci(k: int, hours: float, alpha: float = 0.05) -> tuple[float, float]:
    """Exact (Garwood) confidence interval of an event rate per hour from k events in `hours` of audio."""
    if hours <= 0:
        return float("nan"), float("nan")
    lo = 0.0 if k == 0 else stats.chi2.ppf(alpha / 2, 2 * k) / 2.0
    hi = stats.chi2.ppf(1 - alpha / 2, 2 * k + 2) / 2.0
    return float(lo / hours), float(hi / hours)


# ---- sweep and operating point -------------------------------------------------------------------------------------------------

@dataclass
class BgStream:
    source: str                    # "own" or "esc50_crossfit"
    obs: list[al.Obs]
    hours: float
    segments: list[Segment] | None = None


@dataclass
class ScenarioStream:
    snr_db: float
    obs: list[al.Obs]
    truth: list[dict[str, float]]


def params_for(alert_cfg: Mapping[str, Any], t_on: float, default_t_on: float) -> al.AlertParams:
    return al.AlertParams.from_cfg({**alert_cfg, "t_on": float(t_on), "t_off": None}, default_t_on)


def sweep(bg: Sequence[BgStream], scen: Sequence[ScenarioStream], t_ons: Sequence[float], alert_cfg: Mapping[str, Any], label: str,
          tol_s: float, selection_snrs: Sequence[float] | None = None, default_t_on: float = 0.3) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the alert logic over every stream for each t_on.

    Returns (table, per_snr): `table` has one row per t_on with false alarms per source (count, hours, rate, exact 95% CI) and
    the mean event recall over the selection SNRs; `per_snr` has recall and latency per (t_on, scenario SNR).
    """
    sources = sorted({b.source for b in bg})
    snrs = sorted({s.snr_db for s in scen}, reverse=True)
    sel = list(snrs if selection_snrs is None else [s for s in snrs if s in set(selection_snrs)])
    rows, per = [], []
    for t in t_ons:
        pr = params_for(alert_cfg, t, default_t_on)
        row: dict[str, Any] = {"t_on": float(t), "t_off": pr.t_off}
        for src in sources:
            k, hrs = 0, 0.0
            for b in (b for b in bg if b.source == src):
                k += len(al.detect_events(pr, b.obs, label))
                hrs += b.hours
            lo, hi = poisson_ci(k, hrs)
            row.update({f"fa_count_{src}": k, f"hours_{src}": hrs, f"fa_per_hour_{src}": k / hrs if hrs else float("nan"),
                        f"fa_per_hour_{src}_lo": lo, f"fa_per_hour_{src}_hi": hi})
        rec = {}
        for snr in snrs:
            det, lat, nfalse = [], [], 0
            for s in (s for s in scen if s.snr_db == snr):
                m = match_events(s.truth, al.detect_events(pr, s.obs, label), tol_s)
                det += m["detected"]
                lat += m["latency_s"]
                nfalse += len(m["false_events"])
            lat_ok = [v for v in lat if np.isfinite(v)]
            rec[snr] = float(np.mean(det)) if det else float("nan")
            per.append({"t_on": float(t), "snr_db": snr, "n_clips": len(det), "detected": int(sum(det)), "event_recall": rec[snr],
                        "latency_median_s": float(np.median(lat_ok)) if lat_ok else float("nan"),
                        "latency_p90_s": float(np.percentile(lat_ok, 90)) if lat_ok else float("nan"), "false_events_in_scenarios": nfalse})
        row["event_recall_mean"] = float(np.mean([rec[s] for s in sel])) if sel else float("nan")
        rows.append(row)
    return pd.DataFrame(rows), pd.DataFrame(per)


def choose_operating_point(table: pd.DataFrame, source: str, max_per_hour: float) -> dict[str, Any] | None:
    """Among thresholds with at most `max_per_hour` false alarms per hour on `source`, the one with the best mean event
    recall; ties go to the higher threshold (fewer alarms). None if no threshold qualifies."""
    ok = table[table[f"fa_per_hour_{source}"] <= max_per_hour + 1e-12]
    if ok.empty:
        return None
    best = ok[ok["event_recall_mean"] >= ok["event_recall_mean"].max() - 1e-12].sort_values("t_on", ascending=False).iloc[0]
    return {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in best.to_dict().items()}


def manifest_folds(manifest: pd.DataFrame, ds: Dataset, folds: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    """Fold whose model never saw each manifest row's recording (the fold that held its group out)."""
    from .robustness import gated_fold
    return gated_fold(manifest["group_id"].astype(str).to_numpy(), ds.groups, folds)


def threshold_grid(t_min: float, t_max: float, step: float, extra: Sequence[float] = ()) -> list[float]:
    """Sweep grid rounded to 4 decimals, plus the `extra` values EXACTLY as given (e.g. the model's saved threshold), so a
    lookup of an extra value in the sweep table always finds its row."""
    g = {round(float(t), 4) for t in np.arange(t_min, t_max + 1e-9, step)}
    return sorted(g | {float(e) for e in extra})
