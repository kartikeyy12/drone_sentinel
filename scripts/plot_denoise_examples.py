#!/usr/bin/env python
"""Before/after spectrograms of the denoising variants for one clip per class (drone, aircraft, ambient).

A class with no clips in the manifest (drone, until DADS is configured) is skipped with a message.
Writes results/figures/denoise_examples.png (200 dpi) and results.json key `denoise_examples`.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sentinel.config import PROJECT_ROOT, load_config, project_path, set_seeds, setup_logging, update_results  # noqa: E402
from sentinel.denoise import VARIANTS, denoise, stft  # noqa: E402
from sentinel.preprocess import load_audio, preprocess_clip  # noqa: E402

log = logging.getLogger("plot_denoise_examples")
TITLES = {"none": "band-pass only", "specsub": "spectral subtraction", "wiener": "Wiener gain",
          "svd_energy": "SVD, energy share", "svd_elbow": "SVD, elbow"}
INK, MUTED = "#1f2328", "#59636e"  # neutral text colours; the single-hue ramp carries magnitude
DB_RANGE = 70.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--seed", type=int, default=None, help="clip choice (default: cfg seed)")
    args = ap.parse_args()
    setup_logging()
    cfg = load_config(args.config)
    seed = cfg["seed"] if args.seed is None else args.seed
    set_seeds(seed)
    man_path = project_path(cfg, "manifest")
    if not man_path.exists():
        log.error("%s not found. Run: python scripts/build_manifest.py", man_path)
        return 1
    man = pd.read_csv(man_path)
    rng = np.random.default_rng(seed)
    sr, d = cfg["audio"]["sr"], cfg["denoise"]
    picks = []
    for cls in cfg["classes"]:
        pool = man[(man["label"] == cls) & (~man["hard_negative"]) & (man["duration_s"] >= 2.0)]
        if pool.empty:
            print(f"no {cls} clips in the manifest, skipping that row")
            continue
        picks.append((cls, pool.iloc[int(rng.integers(len(pool)))]))
    if not picks:
        log.error("manifest has no usable clips")
        return 1

    fig, axes = plt.subplots(len(picks), len(VARIANTS), figsize=(3.6 * len(VARIANTS), 2.9 * len(picks)),
                             squeeze=False, constrained_layout=True)
    record: dict = {"seed": seed, "svd_energy_target": d["svd"]["energy"], "svd_block_s": d["svd"]["block_s"],
                    "db_range_shown": DB_RANGE, "clips": {}}
    for r, (cls, row) in enumerate(picks):
        p = Path(row["path"])
        x = preprocess_clip(load_audio(p if p.is_absolute() else PROJECT_ROOT / p, sr), sr, cfg)
        specs, rec = {}, {"path": row["path"], "category": row["category"], "duration_s": float(row["duration_s"])}
        for v in VARIANTS:
            y, info = denoise(x, v, sr, cfg)
            specs[v] = 20 * np.log10(np.abs(stft(y, d["n_fft"], d["hop"])) + 1e-8)
            rec[v] = {"power_change_db": round(float(10 * np.log10(np.mean(y ** 2) / np.mean(x ** 2))), 2)}
            if info:
                rec[v]["mean_rank_kept"] = round(info["mean_k"], 2)
                rec[v]["rank_available"] = info["blocks"][0]["rank"]
        vmax = float(specs["none"].max())
        for c, v in enumerate(VARIANTS):
            ax = axes[r, c]
            im = ax.imshow(specs[v], origin="lower", aspect="auto", cmap="Blues", vmin=vmax - DB_RANGE, vmax=vmax,
                           extent=[0, len(x) / sr, 0, sr / 2000])
            ax.set_title(f"{TITLES[v]}\n{rec[v]['power_change_db']:+.1f} dB power", fontsize=9, color=INK)
            ax.tick_params(labelsize=8, colors=MUTED)
            if c == 0:
                ax.set_ylabel(f"{cls}\n({row['category']})\nfrequency (kHz)", fontsize=9, color=INK)
            if r == len(picks) - 1:
                ax.set_xlabel("time (s)", fontsize=9, color=INK)
        record["clips"][cls] = rec
    fig.colorbar(im, ax=axes, shrink=0.8, pad=0.01).set_label(f"level (dB re max of band-passed clip, {DB_RANGE:.0f} dB shown)",
                                                               fontsize=8, color=INK)
    out = project_path(cfg, "figures") / "denoise_examples.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)
    update_results(cfg, "denoise_examples", record)
    print(f"wrote {out}")
    for cls, rec in record["clips"].items():
        print(f"  {cls}: {rec['path']}  " + "  ".join(f"{v}:{rec[v]['power_change_db']:+.1f}dB" for v in VARIANTS))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
