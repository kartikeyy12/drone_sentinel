"""Bench distance test: detection rate and measured SNR per distance, with the spherical-spreading prediction overlaid.

Record the same clip, played through a speaker at a fixed volume, with the same microphone at 1, 2, 4 and 8 m and save
data/bench/bench_1m.wav, bench_2m.wav, bench_4m.wav, bench_8m.wav (any other bench_<d>m.wav is picked up too). Optional:
data/bench/bench_noise.wav, the room with the speaker silent at the same gain, used as the noise reference for the SNR.

    python scripts/bench_distance.py [--dir data/bench] [--t-on 0.56]

Writes results.json -> bench, results/tables/m7_bench.csv, results/figures/m7_bench.png. With no bench files it writes
{"available": false} and exits 0, so the report says plainly that no bench data exists.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from sentinel import alerting as al
from sentinel import plotstyle as ps
from sentinel.bench import evaluate_file, find_bench_files, inband, predicted_snr_db
from sentinel.bundle import load_model
from sentinel.config import load_config, project_path, setup_logging, update_results
from sentinel.preprocess import load_audio


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=None)
    ap.add_argument("--t-on", type=float, default=None, help="alert threshold on the smoothed probability (default: m5.alert.t_on or the model threshold)")
    ap.add_argument("--config", default=None)
    args = ap.parse_args(argv)
    setup_logging(logging.WARNING)
    cfg = load_config(args.config)
    folder = Path(args.dir) if args.dir else project_path(cfg, "data_bench")
    files = find_bench_files(folder)
    if not files:
        update_results(cfg, "bench", {"available": False, "folder": str(folder),
                                      "note": "No bench_<d>m.wav files found. No bench measurement was made; nothing here is measured."})
        print(f"no bench_<d>m.wav files in {folder}: wrote bench.available = false")
        return 0
    model = load_model(project_path(cfg, "models") / "pipeline.joblib", cfg)
    target = model.targets[0]
    a = dict(cfg["m5"]["alert"])
    if args.t_on is not None:
        a["t_on"], a["t_off"] = args.t_on, None
    params = {target: al.AlertParams.from_cfg(a, model.thresholds[target])}
    noise_path = folder / cfg["m7"]["bench"]["noise_file"]
    noise = inband(load_audio(noise_path, cfg["audio"]["sr"]), cfg) if noise_path.exists() else None
    rows = [{"distance_m": d, **evaluate_file(model, cfg, p, params, noise)} for d, p in files]
    alpha = float(cfg["m7"]["alpha_db_per_m"])
    ref = next((r for r in rows if r["snr_db"] is not None), None)
    for r in rows:
        r["snr_pred_db"] = predicted_snr_db(r["distance_m"], ref["distance_m"], ref["snr_db"], alpha) if ref else None
    df = pd.DataFrame(rows)
    tables = project_path(cfg, "tables")
    tables.mkdir(parents=True, exist_ok=True)
    df.to_csv(tables / "m7_bench.csv", index=False)
    ps.apply()
    fig, ax = ps.plt.subplots(1, 2, figsize=(9, 3.4))
    ax[0].plot(df.distance_m, df.snr_db, label="measured", **ps.style(0))
    ax[0].plot(df.distance_m, df.snr_pred_db, label="spherical spreading", **ps.style(1))
    ax[0].set_xscale("log", base=2)
    ax[0].set_xlabel("distance (m)")
    ax[0].set_ylabel("SNR (dB)")
    ax[0].set_title("Measured SNR vs prediction")
    ax[0].legend()
    ax[1].plot(df.distance_m, df.window_detection_rate, **ps.style(2))
    ax[1].set_xscale("log", base=2)
    ax[1].set_ylim(-0.02, 1.02)
    ax[1].set_xlabel("distance (m)")
    ax[1].set_ylabel("share of windows at or above threshold")
    ax[1].set_title("Window detection rate")
    fig.tight_layout()
    ps.save(fig, project_path(cfg, "figures") / "m7_bench.png")
    update_results(cfg, "bench", {
        "available": True, "folder": str(folder), "target_class": target, "rows": rows, "prediction_anchor_m": ref["distance_m"] if ref else None,
        "alpha_db_per_m": alpha, "noise_file_used": noise is not None,
        "note": ("The recordings are of whatever clip the user played, through a speaker, in the room where the user recorded. The class model is "
                 "aircraft versus ambient (ESC-50), so the detection rate is the aircraft probability, not a drone result. Indoor reverberation breaks "
                 "the free-field assumption of the prediction."),
    })
    print(df.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
