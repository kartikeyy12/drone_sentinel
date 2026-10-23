#!/usr/bin/env python
"""M4 `make robustness`: how much does noise hurt a model trained on clean audio?

For every noise kind and SNR, the raw audio of every recording gets additive noise (see src/sentinel/noise.py), is run
through the full chain (band-pass, denoising variant, features) and is scored by the model of the fold that held its
source recording out. Features are cached per (kind, SNR, variant), so a rerun only computes what is missing.

Also runs the augmentation study (train on clean + the OTHER noise kinds, test on the held-out kind).
Writes results.json `robustness`, results/tables/m4_robustness*.csv and results/figures/m4_robustness_*.png.
Everything here is ESC-50, aircraft versus ambient; nothing is a drone result.
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sentinel import robustness as rb  # noqa: E402
from sentinel.config import PROJECT_ROOT, load_config, project_path, set_seeds, setup_logging, update_results  # noqa: E402
from sentinel.cv import make_folds  # noqa: E402
from sentinel.dataset import build_dataset, gated_ambient_windows  # noqa: E402
from sentinel.m4plots import plot_robustness_ba, plot_robustness_components  # noqa: E402
from sentinel.noise import reference_power  # noqa: E402

log = logging.getLogger("run_robustness")


def flat(df: pd.DataFrame) -> pd.DataFrame:
    folds = pd.DataFrame(df["ba_folds"].tolist(), index=df.index).add_prefix("ba_fold")
    return pd.concat([df.drop(columns=["ba_folds"]), folds], axis=1)


def clip_snr_summary(cfg, manifest, kinds, snrs, variant) -> dict:
    """In-band SNR each clip actually received (the noise level is the same for every clip, the clips are not)."""
    out: dict = {}
    for k in kinds:
        for s in snrs:
            with np.load(rb.noisy_path(cfg, k, s, variant), allow_pickle=False) as z:
                snr = z["clip_snr_db"]
            d = {}
            for lab in ("aircraft", "ambient"):
                v = snr[(manifest["label"] == lab).to_numpy() & np.isfinite(snr)]
                d[lab] = {"n_clips": int(len(v)), "min": float(v.min()), "p10": float(np.percentile(v, 10)),
                          "median": float(np.median(v)), "p90": float(np.percentile(v, 90)), "max": float(v.max()),
                          "share_below_0_db": float(np.mean(v < 0))}
            out.setdefault(k, {})[f"{s:+g}"] = d
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--quick", action="store_true", help="one noise kind, one SNR, variant none, no augmentation (smoke test)")
    ap.add_argument("--force", action="store_true", help="rebuild noisy caches even if fresh")
    ap.add_argument("--n-jobs", type=int, default=None)
    ap.add_argument("--skip-build", action="store_true", help="evaluate existing caches only")
    args = ap.parse_args()
    setup_logging()
    cfg = load_config(args.config)
    set_seeds(cfg["seed"])
    r = cfg["m4"]["robustness"]
    kinds, snrs, variants = list(r["kinds"]), [float(s) for s in r["snr_db"]], list(r["variants"])
    n_boot, aug_on, aug_variants = int(r["n_bootstrap"]), bool(r["augmentation"]["enabled"]), list(r["augmentation"]["variants"])
    tag = ""
    if args.quick:
        kinds, snrs, variants, n_boot, aug_on, tag = kinds[:1], snrs[:1], variants[:1], 100, False, "_quick"
    n_jobs = args.n_jobs if args.n_jobs is not None else cfg["features"]["n_jobs"]
    seed, pos = cfg["seed"], cfg["m4"]["positive_classes"][0]
    t0 = time.perf_counter()

    manifest = pd.read_csv(project_path(cfg, "manifest"))  # the whole manifest: rows must line up with the clean caches
    try:
        ds = {v: build_dataset(cfg, v) for v in variants}
    except FileNotFoundError as e:
        log.error("%s", e)
        return 1
    if len(manifest) != len(np.unique(load_cache_file_index(cfg))):
        log.error("manifest and clean feature caches disagree on the number of files; run make features")
        return 1
    ref_variant = variants[0]
    for v in variants:
        if not (np.array_equal(ds[v].y, ds[ref_variant].y) and np.array_equal(ds[v].groups, ds[ref_variant].groups)):
            log.error("variant %s has different windows than %s", v, ref_variant)
            return 1

    # ---- reference level and noise pool -------------------------------------------------------------------------------
    air = manifest[(manifest["label"] == pos) & ~manifest["dataset"].isin(cfg["splits"]["holdout_datasets"])]
    p_ref = reference_power([str(PROJECT_ROOT / p) for p in air["path"]], cfg["audio"]["sr"], cfg)
    pool = rb.build_noise_pool(cfg, manifest) if "ambient" in kinds else None
    print(f"reference level: median in-band power of {len(air)} {pos} clips = {p_ref:.3e} (amplitude units^2, 16 kHz float)")
    if pool is not None:
        print(f"ambient noise pool: {len(pool)} usable recordings from {r['ambient_noise']['categories']}")

    # ---- noisy feature caches -------------------------------------------------------------------------------------------
    builds = []
    if not args.skip_build:
        for k in kinds:
            b = rb.build_noisy(cfg, manifest, k, snrs, variants, p_ref, pool, n_jobs=n_jobs, force=args.force)
            builds.append(b)
            print(f"[{k:8s}] built {b['built'] or 'nothing (cache hit)'} in {b['wall_seconds']} s", flush=True)

    # ---- evaluation ----------------------------------------------------------------------------------------------------
    folds = make_folds(ds[ref_variant].y, ds[ref_variant].groups, cfg["splits"]["cv_folds"], seed)
    gated = gated_ambient_windows(cfg)
    feats = rb.NoisyFeatures(cfg)
    conditions = [rb.CLEAN] + [(k, s) for k in kinds for s in snrs]
    recs = []
    for v in variants:
        recs += rb.evaluate_conditions(cfg, ds[v], gated, folds, v, feats, conditions, rb.clean_train(ds[v]), n_boot, seed,
                                       "clean_trained")
        print(f"evaluated {v}", flush=True)
    # same features, level (loudness) features dropped: are they what noise breaks?
    nolevel = "none (no level features)"
    if "none" in variants and not cfg["m3"]["default"]["drop_level_features"]:
        c2 = copy.deepcopy(cfg)
        c2["m3"]["default"]["drop_level_features"] = True
        recs += rb.evaluate_conditions(c2, ds["none"], gated, folds, "none", feats, conditions, rb.clean_train(ds["none"]), n_boot,
                                       seed, "clean_trained_no_level_features", label=nolevel)
        print("evaluated none without level features", flush=True)
    R = pd.DataFrame(recs)
    aug_recs = []
    if aug_on and len(kinds) >= 2:
        for v in aug_variants:
            if v not in variants:
                log.warning("augmentation variant %s has no noisy caches (not in m4.robustness.variants); skipped", v)
                continue
            for k in kinds:
                rr = rb.evaluate_conditions(cfg, ds[v], gated, folds, v, feats, [rb.CLEAN] + [(k, s) for s in snrs],
                                            rb.augmented_train(ds[v], feats, v, k, kinds, snrs, seed), n_boot, seed,
                                            "augmented_leave_one_kind_out")
                for x in rr:
                    x["held_out_kind"] = k
                aug_recs += rr
            print(f"evaluated augmentation ({v})", flush=True)
    A = pd.DataFrame(aug_recs)
    paired = rb.paired_against(R, "none") if "none" in variants and (R["variant"] != "none").any() else pd.DataFrame()
    aug_paired = rb.paired_aug_vs_clean(R, A) if len(A) else pd.DataFrame()

    # ---- consistency with M3 -------------------------------------------------------------------------------------------
    check = None
    try:
        cv = json.loads(project_path(cfg, "results_json").read_text())["cv_default"]
        d = cfg["m3"]["default"]
        if cv["config"]["variant"] in variants:
            row = R[(R["variant"] == cv["config"]["variant"]) & (R["kind"] == "clean")].iloc[0]
            m3 = cv["fold_mean_std"]["balanced_accuracy_mean"]
            check = {"variant": cv["config"]["variant"], "robustness_clean_ba_fold_mean": float(row["ba_fold_mean"]),
                     "cv_default_ba_fold_mean": float(m3), "abs_difference": abs(float(row["ba_fold_mean"]) - m3),
                     "same_pipeline_settings": cv["config"]["pca"] == d["pca"] and cv["config"]["svm"] == d["svm"]}
    except (FileNotFoundError, KeyError):
        pass

    # ---- write ---------------------------------------------------------------------------------------------------------
    tables, figs = project_path(cfg, "tables"), project_path(cfg, "figures")
    tables.mkdir(parents=True, exist_ok=True)
    flat(R).to_csv(tables / f"m4_robustness{tag}.csv", index=False)
    if len(paired):
        paired.to_csv(tables / f"m4_robustness_paired{tag}.csv", index=False)
    if len(A):
        flat(A).to_csv(tables / f"m4_robustness_augmentation{tag}.csv", index=False)
        aug_paired.to_csv(tables / f"m4_robustness_augmentation_paired{tag}.csv", index=False)
    shown = variants + ([nolevel] if (R["variant"] == nolevel).any() else [])
    plot_robustness_ba(R, A if len(A) else None, kinds, snrs, shown, figs / f"m4_robustness_ba{tag}.png", pos)
    plot_robustness_components(R, variants[0], kinds, snrs, figs / f"m4_robustness_components{tag}.png", pos)
    update_results(cfg, f"robustness{tag}", {
        "config": {"kinds": kinds, "snr_db": snrs, "variants": variants, "pipeline": cfg["m3"]["default"],
                   "cv": f"StratifiedGroupKFold({cfg['splits']['cv_folds']}, shuffle, seed={seed}) on source group, trained on clean audio",
                   "n_bootstrap": n_boot},
        "reference": {"definition": f"median in-band (100-7000 Hz) power of the {len(air)} {pos} clips; the same absolute noise "
                                    "level is added to every clip", "p_ref": p_ref, "n_reference_clips": int(len(air))},
        "noise_pool": None if pool is None else {"n_recordings": len(pool), **r["ambient_noise"]},
        "windows": {"scored": int(len(ds[ref_variant].y)), "noise_only": int(len(gated.row_id))},
        "clip_snr_db": clip_snr_summary(cfg, manifest, kinds, snrs, variants[0]),
        "clean_matches_cv_default": check, "records": recs, "paired_vs_none": paired.to_dict("records"),
        "augmentation": aug_recs, "augmentation_vs_clean_trained": aug_paired.to_dict("records"),
        "builds": builds, "wall_seconds": round(time.perf_counter() - t0, 1),
        "note": "ESC-50 only, aircraft versus ambient. Noise kinds white and pink are synthetic; 'ambient' uses ESC-50 recordings "
                "of classes the SVM was also trained on as negatives."})

    # ---- print ---------------------------------------------------------------------------------------------------------
    if check:
        print(f"\nconsistency with `make train`: clean balanced accuracy (fold mean) {check['robustness_clean_ba_fold_mean']:.6f} "
              f"vs {check['cv_default_ba_fold_mean']:.6f}, difference {check['abs_difference']:.2e}")
    print("\npooled out-of-fold balanced accuracy [95% group-bootstrap CI], trained on clean audio")
    cols = ["clean"] + [f"{s:+g}" for s in sorted(snrs, reverse=True)]
    rows = []
    for v in shown:
        for k in kinds:
            sel = R[(R["variant"] == v) & ((R["kind"] == k) | (R["kind"] == "clean"))]
            rows.append([v, k] + [f"{x.ba_pooled:.3f} [{x.ba_lo:.2f},{x.ba_hi:.2f}]"
                                  for x in (sel[sel["snr_db"].isna()].iloc[0] if s is None else sel[sel["snr_db"] == s].iloc[0]
                                            for s in [None] + sorted(snrs, reverse=True))])
    print(pd.DataFrame(rows, columns=["variant", "noise"] + cols).to_string(index=False))
    v0 = variants[0]
    print(f"\ncomponents for {v0} (aircraft recall / ambient specificity / hard-negative specificity / AUC / noise-only FAR)")
    for k in [None] + kinds:
        for s in ([None] if k is None else sorted(snrs, reverse=True)):
            x = R[(R["variant"] == v0) & ((R["kind"] == "clean") if k is None else ((R["kind"] == k) & (R["snr_db"] == s)))].iloc[0]
            print(f"  {k or 'clean':8s} {'' if s is None else f'{s:+g} dB':>7s}: recall {x[f'recall_{pos}']:.3f}  spec {x['specificity_plain']:.3f}  "
                  f"hard-neg {x['specificity_hard_negative']:.3f}  AUC {x[f'auc_{pos}']:.3f}  noise-only FAR {x['noise_only_far']:.3f}"
                  f" (n={x['n_noise_only']})")
    if len(paired):
        print("\ndenoising variant minus `none`, paired over folds (mean diff, folds better of 5):")
        print(paired.round(3).to_string(index=False))
    if len(aug_paired):
        print("\naugmented training (clean + other noise kinds) minus clean-only training, on the held-out kind:")
        print(aug_paired[aug_paired["kind"] != "clean"].round(3).to_string(index=False))
    print(f"\nwrote results/tables/m4_robustness{tag}*.csv, results/figures/m4_robustness_*{tag}.png, results.json `robustness{tag}` "
          f"({time.perf_counter() - t0:.0f} s)")
    return 0


def load_cache_file_index(cfg) -> np.ndarray:
    from sentinel.dataset import load_cache
    return load_cache(cfg, "none")["file_index"]


if __name__ == "__main__":
    raise SystemExit(main())
