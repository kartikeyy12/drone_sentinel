"""Detection range from spherical spreading: a model estimate, not a measurement.

Condition for detection at distance d (free field, broadband levels):

    L_src - 20 log10(d / d0) - alpha * d - L_ambient >= snr_min_db

`snr_min_db` is read from a measured recall-versus-SNR curve (lowest SNR from which the recall stays at or above a target).
The source level is never assumed: it is a config parameter (`m7.drone_source_level_db_at_1m`, TODO_CITE).

Simplifications: free field, one broadband SNR (no frequency weighting, no A-weighting), no wind, no ground effect, no
reverberation, a constant ambient level, alpha constant over the path.
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np
from scipy import optimize

SIMPLIFICATIONS = [
    "free field, spherical spreading from a point source (no ground reflection, no ground effect)",
    "one broadband SNR; no frequency weighting and no A-weighting",
    "no wind or temperature gradients, no refraction, no turbulence",
    "no reverberation (indoor bench tests break this assumption)",
    "constant ambient level and constant atmospheric absorption along the path",
    "the SNR threshold comes from ESC-50 aircraft clips, not from drone recordings",
]


def margin_db(d_m: float, source_level_db: float, ambient_level_db: float, snr_min_db: float, alpha_db_per_m: float = 0.0,
              d0_m: float = 1.0) -> float:
    """Left-hand side minus right-hand side of the detection condition at distance d (positive = detected)."""
    return source_level_db - 20.0 * math.log10(d_m / d0_m) - alpha_db_per_m * d_m - ambient_level_db - snr_min_db


def range_m(source_level_db: float, ambient_level_db: float, snr_min_db: float, alpha_db_per_m: float = 0.0,
            d0_m: float = 1.0) -> float | None:
    """Largest distance d >= d0 that still meets the condition. None if it is not met even at d0.

    alpha = 0 has a closed form; otherwise the (strictly decreasing) margin is solved by bracketing and Brent's method.
    """
    if alpha_db_per_m < 0:
        raise ValueError("alpha must not be negative")
    if d0_m <= 0:
        raise ValueError("d0 must be positive")
    excess = source_level_db - ambient_level_db - snr_min_db          # margin in dB at d = d0 when alpha = 0
    if margin_db(d0_m, source_level_db, ambient_level_db, snr_min_db, alpha_db_per_m, d0_m) < 0:
        return None
    if alpha_db_per_m == 0.0:
        return float(d0_m * 10.0 ** (excess / 20.0))
    hi = d0_m
    while margin_db(hi, source_level_db, ambient_level_db, snr_min_db, alpha_db_per_m, d0_m) > 0:
        hi *= 2.0
        if hi > 1e9:
            return float("inf")
    return float(optimize.brentq(lambda d: margin_db(d, source_level_db, ambient_level_db, snr_min_db, alpha_db_per_m, d0_m),
                                 d0_m, hi, xtol=1e-9, rtol=1e-12))


def snr_min_from_curve(snr_db: Sequence[float], recall: Sequence[float], target: float) -> float | None:
    """Lowest tested SNR from which the recall stays at or above `target` at that SNR and at every higher tested SNR.

    None when the highest tested SNR already misses the target ("not reached"). Requiring every higher point to pass keeps one
    lucky low-SNR point from defining the threshold.
    """
    order = np.argsort(np.asarray(snr_db, float))[::-1]
    best = None
    for i in order:
        r = float(recall[i])
        if not math.isfinite(r) or r < target:
            break
        best = float(snr_db[i])
    return best


def sensitivity_table(source_levels_db: Sequence[float], ambient_levels_db: Sequence[float], snr_min_values_db: Sequence[float],
                      alpha_db_per_m: float = 0.0, d0_m: float = 1.0) -> list[dict[str, Any]]:
    """Range for every (required SNR, source level, ambient level); range_m is None where it is below d0."""
    rows = []
    for s in snr_min_values_db:
        for ls in source_levels_db:
            for la in ambient_levels_db:
                rows.append({"snr_min_db": float(s), "source_level_db": float(ls), "ambient_level_db": float(la),
                             "range_m": range_m(ls, la, s, alpha_db_per_m, d0_m)})
    return rows


def _parse_snr(key: str) -> float:
    return float(key)


def curves_from_results(res: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Recall-versus-SNR curves stored in results.json by M4 and M6.

    - m6_event_recall_at_operating_point / m6_event_recall_at_model_threshold: event recall of embedded aircraft clips; the SNR is
      the clip's in-band power over the background stream's in-band power (the same broadband idea as the range model).
    - m4_window_recall_<noise kind>: window recall of the SVM boundary (native threshold, not the operating point) with noise
      of a fixed absolute level; the SNR is nominal against the median aircraft clip.
    Each curve: {"snr_db": [...], "recall": [...], "axis": text, "unit": "event" | "window"}.
    """
    out: dict[str, dict[str, Any]] = {}
    fa = res.get("false_alarms", {})
    for name, key in (("m6_event_recall_at_operating_point", "at_operating_point"), ("m6_event_recall_at_model_threshold", "at_model_threshold")):
        by = fa.get(key, {}).get("by_snr")
        if by:
            snrs = sorted(((_parse_snr(k), v["event_recall"]) for k, v in by.items()), reverse=True)
            out[name] = {"snr_db": [s for s, _ in snrs], "recall": [r for _, r in snrs], "unit": "event",
                         "axis": "clip in-band power over background stream in-band power (M6)"}
    rec = res.get("robustness", {}).get("records", [])
    for kind in ("white", "pink", "ambient"):
        pts = sorted(((r["snr_db"], r["recall_aircraft"]) for r in rec
                      if r.get("tag") == "clean_trained" and r.get("variant") == "none" and r.get("kind") == kind and r.get("snr_db") is not None),
                     reverse=True)
        if pts:
            out[f"m4_window_recall_{kind}"] = {"snr_db": [float(s) for s, _ in pts], "recall": [float(r) for _, r in pts], "unit": "window",
                                              "axis": "nominal SNR against the median aircraft clip, noise at a fixed absolute level (M4)"}
    return out


def analyse(res: Mapping[str, Any], cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Everything the range section of the report needs, from results.json and the m7 config. Writes nothing."""
    m7 = cfg["m7"]
    curves = curves_from_results(res)
    target = float(m7["recall_target"])
    primary_name = m7["primary_curve"]
    if primary_name not in curves:
        raise KeyError(f"primary curve {primary_name!r} not in results.json; available: {sorted(curves)}")
    primary = curves[primary_name]
    snr_min = snr_min_from_curve(primary["snr_db"], primary["recall"], target)
    reported = {}
    for name, c in curves.items():
        reported[name] = {"max_recall": float(max(c["recall"])), "snr_at_max_recall_db": float(c["snr_db"][int(np.argmax(c["recall"]))]),
                          "snr_min_db_for_target": {f"{t:g}": snr_min_from_curve(c["snr_db"], c["recall"], float(t))
                                                    for t in m7["recall_targets_reported"]}}
    sens = m7["sensitivity"]
    alpha, d0 = float(m7["alpha_db_per_m"]), float(m7["d0_m"])
    hypo = [float(s) for s in sens["hypothetical_snr_min_db"]]
    table = sensitivity_table(sens["source_levels_db"], sens["ambient_levels_db"], hypo, alpha, d0)
    ls, la = m7["drone_source_level_db_at_1m"], m7["ambient_level_db_site"]
    missing = []
    if snr_min is None:
        missing.append(f"snr_min_db: the {primary_name} curve never reaches recall {target:g} (best {max(primary['recall']):.3f})")
    if ls is None:
        missing.append("drone_source_level_db_at_1m is not set (TODO_CITE)")
    if la is None:
        missing.append("ambient_level_db_site is not set")
    estimate = None
    if not missing:
        estimate = range_m(float(ls), float(la), snr_min, alpha, d0)
    return {
        "recall_target": target, "primary_curve": primary_name, "primary_curve_axis": primary["axis"],
        "snr_min_db": snr_min, "snr_min_reached": snr_min is not None,
        "range_estimate_m": estimate, "range_estimate_missing_inputs": missing,
        "inputs": {"drone_source_level_db_at_1m": ls, "ambient_level_db_site": la, "alpha_db_per_m": alpha, "d0_m": d0},
        "curves": curves, "curve_summary": reported,
        "sensitivity_hypothetical_snr_min_db": hypo, "sensitivity": table,
        "simplifications": SIMPLIFICATIONS,
        "note": ("Model estimate, not a measurement. The required SNR comes from ESC-50 aircraft clips (there is no drone class in the data) and "
                 "the source level is not assumed. The sensitivity table shows the range for hypothetical required SNRs, not for the measured one."),
    }
