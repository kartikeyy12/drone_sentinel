"""Bench distance test: recordings of one clip played at a fixed volume, recorded at several distances.

For every `bench_<d>m.wav` the live engine runs offline and two things are measured: the detection rate and the SNR of the
recording. The prediction is spherical spreading anchored on the SNR measured at the nearest distance, so it needs no source level:

    SNR_pred(d) = SNR_meas(d_ref) - 20 log10(d / d_ref) - alpha * (d - d_ref)

Indoor rooms are reverberant: the direct sound falls by 6 dB per doubling but the reverberant field does not, so a measured
curve flatter than the prediction is expected indoors. The SNR here is the recording's mean frame power over a noise floor
(a silent-speaker recording if given, otherwise the 10th percentile of the recording's own frame power), after the band-pass.
Without a silent-speaker recording, start every recording with a few seconds of silence before the clip plays: a recording that
is loud from the first to the last frame has no noise-only frames, its own-percentile floor sits at the signal level and the SNR
is meaningless (it is then reported with `noise_floor_is_own_percentile` true and should not be trusted).
The detection rate is taken over the playback windows: windows whose own SNR estimate is at least `m7.bench.active_snr_db`.
"""
from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from . import alerting as al
from .engine import run_offline
from .preprocess import bandpass_offline, bandpass_sos, load_audio, remove_dc

BENCH_RE = re.compile(r"^bench_(\d+(?:\.\d+)?)m\.wav$")


def find_bench_files(folder: Path) -> list[tuple[float, Path]]:
    """(distance in metres, path) of every bench_<d>m.wav in the folder, sorted by distance."""
    out = []
    for p in Path(folder).glob("bench_*m.wav"):
        m = BENCH_RE.match(p.name)
        if m:
            out.append((float(m.group(1)), p))
    return sorted(out)


def predicted_snr_db(d_m: float, ref_d_m: float, ref_snr_db: float, alpha_db_per_m: float = 0.0) -> float:
    """Spherical-spreading prediction anchored at one measured point."""
    return float(ref_snr_db - 20.0 * math.log10(d_m / ref_d_m) - alpha_db_per_m * (d_m - ref_d_m))


def inband(x: np.ndarray, cfg: Mapping[str, Any]) -> np.ndarray:
    bp = cfg["preprocess"]["bandpass"]
    sr = cfg["audio"]["sr"]
    return bandpass_offline(remove_dc(x), bandpass_sos(sr, bp["low_hz"], bp["high_hz"], bp["order"]))


def measure_snr_db(x_inband: np.ndarray, cfg: Mapping[str, Any], noise_inband: np.ndarray | None = None) -> dict[str, Any]:
    """SNR of an in-band recording: (P - N) / N with P the mean frame power and N the noise power.

    N is the mean frame power of `noise_inband` when given, else the percentile (m5.snr.percentile) of the recording's own frame
    powers. Returns the dB value (None if N is not usable) and which noise reference was used.
    """
    sr = cfg["audio"]["sr"]
    flen = int(round(cfg["m5"]["snr"]["frame_s"] * sr))
    p = al.frame_powers(x_inband, flen)
    if noise_inband is not None and len(noise_inband) >= flen:
        n = float(np.mean(al.frame_powers(noise_inband, flen)))
        ref = "silent-speaker recording"
    else:
        n = al.noise_floor(p, cfg["m5"]["snr"]["percentile"])
        ref = f"{cfg['m5']['snr']['percentile']}th percentile of the recording's own frame power"
    return {"snr_db": al.snr_db(float(np.mean(p)), n), "noise_reference": ref, "mean_power": float(np.mean(p)), "noise_power": n,
            "own_percentile": not (noise_inband is not None and len(noise_inband) >= flen)}


def evaluate_file(model, cfg: Mapping[str, Any], path: Path, params=None, noise_inband: np.ndarray | None = None) -> dict[str, Any]:
    """Run one bench recording through the engine and measure it. Detection rate = share of scored windows at or above the threshold."""
    x = load_audio(path, cfg["audio"]["sr"])
    wins, recs, eng = run_offline(model, cfg, x, params=params)
    target = model.targets[0]
    thr = (params[target].t_on if params else model.thresholds[target])
    scored = [w for w in wins if not w.gated]
    hit_all = sum(1 for w in scored if w.p[target] >= thr)
    act_snr = float(cfg["m7"]["bench"]["active_snr_db"])
    active = [w for w in scored if w.snr_db is not None and w.snr_db >= act_snr]
    hit = sum(1 for w in active if w.p[target] >= thr)
    snr = measure_snr_db(inband(x, cfg), cfg, noise_inband)
    return {"file": path.name, "seconds": round(len(x) / cfg["audio"]["sr"], 3), "n_windows": len(wins), "n_gated": len(wins) - len(scored),
            "n_active_windows": len(active), "active_snr_db": act_snr,
            "window_detection_rate": (hit / len(active)) if active else None, "windows_at_or_above_threshold": hit,
            "window_detection_rate_all_windows": (hit_all / len(scored)) if scored else None, "t_on_used": float(thr),
            "n_events": len(recs), "event_seconds": float(sum(r["duration_s"] for r in recs)), "mean_p": float(np.mean([w.p[target] for w in scored])) if scored else None,
            "snr_db": snr["snr_db"], "noise_reference": snr["noise_reference"],
            "noise_floor_is_own_percentile": snr["own_percentile"]}
