#!/usr/bin/env python
"""Extract per-window features for each denoising variant and cache them as .npz.

Output: data/features/features_<variant>.npz (or ..._smoke_n<N>.npz with --limit-per-class).
A cache is reused when its key (config sections used, variant, manifest paths/labels/groups) matches.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sentinel.config import PROJECT_ROOT, load_config, project_path, set_seeds, setup_logging, update_results  # noqa: E402
from sentinel.denoise import VARIANTS  # noqa: E402
from sentinel.features import build_feature_cache  # noqa: E402

log = logging.getLogger("extract_features")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variants", nargs="+", default=None, choices=VARIANTS, help="default: cfg['variants']")
    ap.add_argument("--limit-per-class", type=int, default=None, help="smoke test on a random subset per class")
    ap.add_argument("--n-jobs", type=int, default=None)
    ap.add_argument("--force", action="store_true", help="ignore existing caches")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    setup_logging()
    cfg = load_config(args.config)
    set_seeds(cfg["seed"])
    man_path = project_path(cfg, "manifest")
    if not man_path.exists():
        log.error("%s not found. Run: python scripts/build_manifest.py", man_path)
        return 1
    manifest = pd.read_csv(man_path)
    tag = ""
    if args.limit_per_class:
        keep = [i for _, g in manifest.groupby("label")
                for i in g.sample(min(args.limit_per_class, len(g)), random_state=cfg["seed"]).index]
        manifest = manifest.loc[keep].sort_values("path").reset_index(drop=True)
        tag = f"_smoke_n{args.limit_per_class}"
    out_dir = PROJECT_ROOT / cfg["features"]["cache_dir"]
    n_jobs = args.n_jobs if args.n_jobs is not None else cfg["features"]["n_jobs"]
    print(f"{len(manifest)} files, labels {manifest['label'].value_counts().to_dict()}, n_jobs={n_jobs}")
    for v in args.variants or cfg["variants"]:
        s = build_feature_cache(cfg, manifest, v, out_dir / f"features_{v}{tag}.npz", n_jobs=n_jobs, force=args.force)
        update_results(cfg, "features", {f"{v}{tag}": s}, merge=True)
        print(f"[{v:10s}] {s['cache']:5s} windows={s['n_windows']} by_class={s['windows_by_class']} "
              f"no_full_window={s['files_without_a_full_window']} denoise_rtf={s['denoise_rtf']} "
              f"feat_ms/window={s['feature_ms_per_window']} wall={s['wall_seconds']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
