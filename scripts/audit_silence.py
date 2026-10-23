#!/usr/bin/env python
"""Count near-silent windows per class in the feature cache (variant `none`).

Padded clips (digital silence) yield windows with no sound that still carry the clip's label.
Writes results/tables/silence_audit.csv and results.json key `silence_audit`.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sentinel.config import PROJECT_ROOT, load_config, project_path, update_results  # noqa: E402

THRESHOLDS = {"rms_below_1e-4": 1e-4, "rms_below_1e-3": 1e-3}  # amplitude, full scale = 1.0 (-80 and -60 dBFS)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variant", default="none")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    path = PROJECT_ROOT / cfg["features"]["cache_dir"] / f"features_{args.variant}.npz"
    if not path.exists():
        print(f"{path} not found. Run: python scripts/extract_features.py")
        return 1
    z = np.load(path)
    rms = z["X"][:, list(z["feature_names"]).index("rms_mean")]
    df = pd.DataFrame({"bucket": np.where(z["hard_negative"], "ambient_hard_negative", z["label"]),
                       "file": z["file_index"], "rms": rms})
    rows = []
    for bucket, g in df.groupby("bucket"):
        row = {"bucket": bucket, "windows": int(len(g)), "files": int(g["file"].nunique())}
        for name, thr in THRESHOLDS.items():
            silent = g["rms"] < thr
            row[name] = int(silent.sum())
            row[name + "_pct"] = round(100 * float(silent.mean()), 3)
            per_file = silent.groupby(g["file"]).mean()
            row[name + "_files_half_or_more"] = int((per_file >= 0.5).sum())
        rows.append(row)
    out = pd.DataFrame(rows)
    tab = project_path(cfg, "tables") / "silence_audit.csv"
    tab.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(tab, index=False)
    update_results(cfg, "silence_audit", {"variant": args.variant, "thresholds": THRESHOLDS,
                                          "rows": out.to_dict(orient="records")})
    print(out.to_string(index=False))
    print(f"wrote {tab}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
