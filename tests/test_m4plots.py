"""M4 figures render from hand-built tables."""
from __future__ import annotations

import numpy as np
import pandas as pd

from sentinel import m4plots as mp


def _recs():
    rows = []
    for v in ("none", "wiener"):
        rows.append({"variant": v, "kind": "clean", "snr_db": np.nan, "ba_pooled": 0.79, "ba_lo": 0.73, "ba_hi": 0.84})
        for kind in ("white", "pink"):
            for s, d in ((15.0, 0.05), (-5.0, 0.2)):
                rows.append({"variant": v, "kind": kind, "snr_db": s, "ba_pooled": 0.79 - d, "ba_lo": 0.7 - d, "ba_hi": 0.8 - d,
                             "recall_aircraft": 0.7 - d, "specificity_plain": 0.9, "specificity_hard_negative": 0.7,
                             "auc_aircraft": 0.85, "noise_only_far": 0.0})
    r = pd.DataFrame(rows)
    for c in ("recall_aircraft", "specificity_plain", "specificity_hard_negative", "auc_aircraft", "noise_only_far"):
        r.loc[r["kind"] == "clean", c] = 0.8
    return r


def test_robustness_figures_are_written(tmp_path):
    r = _recs()
    aug = r[r["variant"] == "none"].assign(held_out_kind=lambda d: np.where(d["kind"] == "clean", "white", d["kind"]))
    aug = pd.concat([aug[aug["kind"].isin(["clean", "white"])]])
    p1 = mp.plot_robustness_ba(r, aug, ["white", "pink"], [15.0, -5.0], ["none", "wiener"], tmp_path / "a.png", "aircraft")
    p2 = mp.plot_robustness_components(r, "none", ["white", "pink"], [15.0, -5.0], tmp_path / "b.png", "aircraft")
    assert p1.stat().st_size > 5000 and p2.stat().st_size > 5000
    p3 = mp.plot_robustness_ba(r, None, ["white", "pink"], [15.0, -5.0], ["none", "wiener"], tmp_path / "c.png", "aircraft")
    assert p3.exists()


def test_calibration_and_operating_figures_are_written(tmp_path):
    bins = pd.DataFrame({"kind": ["a"] * 4 + ["b"] * 4, "mean_predicted": [0.01, 0.05, 0.2, 0.5] * 2,
                         "observed": [0.012, 0.04, 0.22, 0.45, 0.001, 0.01, 0.05, 0.3], "n": 10, "bin": list(range(4)) * 2})
    assert mp.plot_calibration(bins, tmp_path / "c.png", "t").stat().st_size > 5000
    curve = pd.DataFrame({"far": np.linspace(0, 1, 50), "recall": np.sqrt(np.linspace(0, 1, 50)), "threshold": 0})
    pts = pd.DataFrame({"target_far": [0.05, 0.1], "far_mean": [0.05, 0.1], "far_std": [0.01, 0.01], "recall_mean": [0.4, 0.6],
                        "recall_std": [0.1, 0.1]})
    assert mp.plot_operating_curve(curve, pts, tmp_path / "o.png", "aircraft", "t").stat().st_size > 5000
