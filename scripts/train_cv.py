#!/usr/bin/env python
"""M3 `make train`: grouped cross-validation of the configured default pipeline.

Reports per-fold and pooled metrics, a group-bootstrap confidence interval, a majority-class baseline and a
label-permutation sanity check (labels shuffled across source recordings; a leak-free pipeline falls to about 0.5
balanced accuracy). Writes results.json `cv_default`, results/tables/m3_oof_default.csv,
m3_false_alarm_by_category_default.csv and results/figures/m3_confusion_default.png.

The default pipeline is NOT tuned. Selection among configurations is done by `make ablation`.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sentinel.config import load_config, project_path, set_seeds, setup_logging, update_results  # noqa: E402
from sentinel.cv import AMBIENT, group_bootstrap_ci, make_folds  # noqa: E402
from sentinel.dataset import build_dataset  # noqa: E402
from sentinel.m3 import cv_run, false_alarm_table, permuted_labels, plot_confusion, pooled_metrics, select_columns  # noqa: E402

log = logging.getLogger("train_cv")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--variant", default=None, help="override m3.default.variant")
    args = ap.parse_args()
    setup_logging()
    cfg = load_config(args.config)
    set_seeds(cfg["seed"])
    m3 = cfg["m3"]
    d = m3["default"]
    variant = args.variant or d["variant"]
    try:
        ds = build_dataset(cfg, variant)
    except FileNotFoundError as e:
        log.error("%s", e)
        return 1
    seed, n_splits = cfg["seed"], cfg["splits"]["cv_folds"]
    t0 = time.perf_counter()
    X = select_columns(ds, cfg, d["drop_level_features"])
    folds = make_folds(ds.y, ds.groups, n_splits, seed)
    classes = ds.classes
    print(f"variant {variant}: {len(ds.y)} windows used, {len(set(ds.groups))} source groups, classes {classes}")
    print(f"  gate: {ds.info['n_gated_ambient']} near-silent ambient windows not shown to the SVM, "
          f"{ds.info['n_dropped_silent_target']} near-silent target-class windows dropped as label noise, "
          f"{ds.info['n_holdout']} holdout windows excluded")
    for c in classes:
        print(f"  {c}: {int(np.sum(ds.y == c))} windows, {len(set(ds.groups[ds.y == c]))} source groups")
    print(f"  pipeline: features {'without' if d['drop_level_features'] else 'with'} level features ({X.shape[1]}), "
          f"PCA {d['pca']}, SVM {d['svm']}, class_weight {m3['class_weight']}\n")

    run = cv_run(X, ds, folds, d["pca"], d["svm"], m3["class_weight"], seed)
    agg, pooled = run["aggregate"], pooled_metrics(ds, run["pred"], run["scores"], run["classes"])
    ci = group_bootstrap_ci(ds.y, run["pred"], ds.groups, classes, m3["bootstrap"]["n"], seed)

    # majority-class baseline through the same folds, and a label-permutation sanity check
    maj = np.empty(len(ds.y), dtype=object)
    for tr, te in folds:
        vals, cnt = np.unique(ds.y[tr], return_counts=True)
        maj[te] = vals[np.argmax(cnt)]
    base = float(np.mean([np.mean(maj[ds.y == c] == c) for c in classes]))
    rng = np.random.default_rng([seed, 3])
    perm = []
    for r in range(m3["permutation_checks"]):
        yp = permuted_labels(ds.y, ds.groups, rng)
        pf = make_folds(yp, ds.groups, n_splits, seed + 1 + r)
        pr = cv_run(X, ds, pf, d["pca"], d["svm"], m3["class_weight"], seed, y=yp)
        perm.append(pr["aggregate"]["balanced_accuracy_mean"])

    per_fold = []
    for i, ((tr, te), m) in enumerate(zip(folds, run["folds"])):
        per_fold.append({"fold": i, "n_train": int(len(tr)), "n_test": int(len(te)),
                         **{f"test_groups_{c}": int(len(set(ds.groups[te][ds.y[te] == c]))) for c in classes},
                         "balanced_accuracy": m["balanced_accuracy"], **{f"recall_{c}": m[f"recall_{c}"] for c in classes}})
    tables, figs = project_path(cfg, "tables"), project_path(cfg, "figures")
    tables.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"row_id": ds.row_id, "group": ds.groups, "category": ds.category, "hard_negative": ds.hard_negative,
                  "label": ds.y, "pred": run["pred"], "fold": run["fold_id"],
                  **{f"score_{c}": run["scores"][:, j] for j, c in enumerate(run["classes"])}}).to_csv(tables / "m3_oof_default.csv", index=False)
    fa = false_alarm_table(ds, run["pred"])
    fa.to_csv(tables / "m3_false_alarm_by_category_default.csv", index=False)
    plot_confusion(ds.y, run["pred"], classes, figs / "m3_confusion_default.png",
                   f"default pipeline, pooled out-of-fold ({variant})")

    update_results(cfg, "cv_default", {
        "config": {"variant": variant, **{k: d[k] for k in ("drop_level_features", "pca", "svm")}, "class_weight": m3["class_weight"],
                   "n_features": int(X.shape[1]), "cv": f"StratifiedGroupKFold({n_splits}, shuffle, seed={seed}) on source group"},
        "dataset": {**ds.info, "classes": classes, "n_groups": int(len(set(ds.groups))),
                    "windows": {c: int(np.sum(ds.y == c)) for c in classes},
                    "groups": {c: int(len(set(ds.groups[ds.y == c]))) for c in classes}},
        "per_fold": per_fold, "fold_mean_std": agg, "pooled_oof": pooled, "pooled_bootstrap_ci95": ci,
        "baseline_majority_class_balanced_accuracy": base,
        "label_permutation_balanced_accuracy": perm,
        "note": "default pipeline is untuned; this is a cross-validated estimate on ESC-50 only, not a drone result",
        "wall_seconds": round(time.perf_counter() - t0, 1)})

    ba, sd = agg["balanced_accuracy_mean"], agg["balanced_accuracy_std"]
    print(f"balanced accuracy  fold mean {ba:.3f} +/- {sd:.3f} (std over folds), pooled {pooled['balanced_accuracy']:.3f}, "
          f"95% group-bootstrap CI {ci['balanced_accuracy_lo']:.3f} to {ci['balanced_accuracy_hi']:.3f}")
    for c in classes:
        print(f"  recall {c:9s} {agg[f'recall_{c}_mean']:.3f} +/- {agg[f'recall_{c}_std']:.3f}   "
              f"(95% CI {ci[f'recall_{c}_lo']:.3f} to {ci[f'recall_{c}_hi']:.3f})")
    for c in classes:
        if c != AMBIENT:
            print(f"  {c}: precision {agg[f'precision_{c}_mean']:.3f}  F1 {agg[f'f1_{c}_mean']:.3f}  AP {agg[f'ap_{c}_mean']:.3f}  "
                  f"AUC {agg[f'auc_{c}_mean']:.3f}")
    print(f"  ambient specificity: plain {agg['specificity_plain_mean']:.3f}, hard negatives {agg['specificity_hard_negative_mean']:.3f}")
    print(f"majority-class baseline balanced accuracy {base:.3f}")
    print("label-permutation check (expect about 0.5): " + ", ".join(f"{p:.3f}" for p in perm))
    print("\nfolds:")
    print(pd.DataFrame(per_fold).round(3).to_string(index=False))
    print("\nhighest false-alarm categories (ambient windows called a target class, pooled out-of-fold):")
    print(fa.head(8).round(3).to_string(index=False))
    print(f"\nwrote results/tables/m3_oof_default.csv, m3_false_alarm_by_category_default.csv, results/figures/m3_confusion_default.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
