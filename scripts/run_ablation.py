#!/usr/bin/env python
"""M3 `make ablation`: full grid, nested selection and feature-group study, all under grouped cross-validation.

1. Grid: denoising variant x level features kept/dropped x PCA size (or none) x SVM setting, on the SAME 5 grouped folds.
   Resumable: finished (configuration, fold) cells are stored in results/tables/m3_grid_folds.jsonl and skipped on rerun.
2. Factor effects: paired differences (same configuration, one factor changed) with win counts.
3. Nested selection: in each outer fold the configuration is chosen by inner grouped CV on the outer training part only,
   then scored once on the outer test part. This is the honest estimate; the best grid cell is optimistic because it was
   picked on the data it is scored on, and the gap between the two is reported.
4. Feature groups: one group left out / kept alone, fixed pipeline.

Everything is ESC-50 (aircraft vs ambient) until drone data exists. Numbers go to results.json `ablation`.
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
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sentinel import plotstyle  # noqa: E402
from sentinel.ablation import (append_rows, block_key, cfg_id, expand_grid, factor_effects, fingerprint, fold_worker,  # noqa: E402
                               load_rows, missing_by_block_fold, nested_outer_worker, subset_worker, summarize_grid, KEY_COLS)
from sentinel.config import PROJECT_ROOT, load_config, project_path, set_seeds, setup_logging, update_results  # noqa: E402
from sentinel.cv import aggregate, group_bootstrap_ci, make_folds  # noqa: E402
from sentinel.dataset import build_dataset, feature_group_masks  # noqa: E402
from sentinel.m3 import false_alarm_table, plot_confusion, select_columns  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

log = logging.getLogger("run_ablation")
VARIANT_LABELS = {"none": "band-pass only", "specsub": "spectral subtraction", "wiener": "Wiener", "svd_energy": "SVD energy",
                  "svd_elbow": "SVD elbow"}


def records(df: pd.DataFrame, nd: int = 4) -> list[dict]:
    """JSON-safe records (NA becomes null, numpy scalars become Python numbers)."""
    return json.loads(df.round(nd).to_json(orient="records"))


def quick(cfg: dict) -> dict:
    """Tiny grid for a smoke test: one variant, two PCA settings, two SVMs."""
    c = copy.deepcopy(cfg)
    g = c["m3"]["grid"]
    g["variants"], g["pca"], g["svm"] = g["variants"][:1], g["pca"][:2], g["svm"][:2]
    return c


def plot_ablation(summary: pd.DataFrame, path: Path) -> None:
    plotstyle.apply()
    s = summary.copy()
    s["pca_key"] = s["pca"].map(lambda v: "none" if pd.isna(v) else str(int(v)))
    s["svm_key"] = s["kernel"] + ", C=" + s["C"].map(lambda v: f"{v:g}")
    s["level_key"] = s["drop_level"].map({False: "kept", True: "dropped"})
    panels = [("variant", "denoising variant", [v for v in VARIANT_LABELS if v in set(s["variant"])]),
              ("level_key", "level (loudness) features", ["kept", "dropped"]),
              ("pca_key", "PCA components", [k for k in ["none", "10", "20", "40", "80"] if k in set(s["pca_key"])]),
              ("svm_key", "SVM", list(dict.fromkeys(s["svm_key"])))]
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.8), sharey=True, constrained_layout=True)
    for ax, (col, title, levels) in zip(axes, panels):
        for i, lev in enumerate(levels):
            v = s[s[col] == lev]["balanced_accuracy_mean"]
            ax.vlines(i, v.min(), v.max(), color=plotstyle.GRID, lw=5, zorder=1)
            ax.plot(i, v.mean(), label="mean over the other factors" if i == 0 else None, zorder=3,
                    **{**plotstyle.style(0), "linestyle": "none"})
            ax.plot(i, v.max(), label="best over the other factors" if i == 0 else None, zorder=3,
                    **{**plotstyle.style(1), "linestyle": "none"})
        labels = [VARIANT_LABELS[l] for l in levels] if col == "variant" else ["no PCA" if l == "none" else l for l in levels]
        ax.set_xticks(range(len(levels)), labels, rotation=20, ha="right")
        ax.set_title(title)
        ax.set_xlim(-0.6, len(levels) - 0.4)
    axes[0].set_ylabel("balanced accuracy (grouped CV, fold mean)")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="outside lower center", ncol=2)
    plotstyle.save(fig, path)


def plot_groups(groups: pd.DataFrame, full: float, path: Path) -> None:
    plotstyle.apply()
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharex=True, constrained_layout=True)
    for ax, kind, title in zip(axes, ("without", "only"), ("one group left out", "one group alone")):
        d = groups[groups["subset"].str.startswith(kind + ":")].copy()
        d["name"] = d["subset"].str.split(":", n=1).str[1]
        d = d.sort_values("balanced_accuracy_mean")
        y = np.arange(len(d))
        ax.errorbar(d["balanced_accuracy_mean"], y, xerr=d["balanced_accuracy_std"], fmt="o", capsize=3, elinewidth=1,
                    **{k: v for k, v in plotstyle.style(0).items() if k not in ("linestyle", "marker")})
        ax.axvline(full, color=plotstyle.MUTED, ls="--", lw=1)
        ax.text(full, -0.45, " all features", fontsize=7, color=plotstyle.MUTED, va="bottom")
        ax.set_ylim(-0.6, len(d) - 0.4)
        ax.set_yticks(y, [f"{n} ({int(k)})" for n, k in zip(d["name"], d["n_features"])])
        ax.set_xlabel("balanced accuracy (grouped CV; bars = std over folds)")
        ax.set_title(title)
    plotstyle.save(fig, path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--force", action="store_true", help="discard stored grid rows and the nested cache")
    ap.add_argument("--skip-nested", action="store_true")
    ap.add_argument("--quick", action="store_true", help="tiny grid for a smoke test; results go to `ablation_quick`")
    args = ap.parse_args()
    setup_logging()
    cfg = load_config(args.config)
    if args.quick:
        cfg = quick(cfg)
    set_seeds(cfg["seed"])
    m3, seed = cfg["m3"], cfg["seed"]
    cw, n_splits = m3["class_weight"], cfg["splits"]["cv_folds"]
    tables, figs = project_path(cfg, "tables"), project_path(cfg, "figures")
    tables.mkdir(parents=True, exist_ok=True)
    tag = "_quick" if args.quick else ""
    t_start = time.perf_counter()

    # ---- data: every variant on identical windows, one set of folds for everything
    variants = m3["grid"]["variants"]
    dsets = {}
    for v in variants:
        try:
            dsets[v] = build_dataset(cfg, v)
        except (FileNotFoundError, ValueError) as e:
            log.error("%s", e)
            return 1
    ref = dsets[variants[0]]
    for v in variants[1:]:
        if not np.array_equal(ref.row_id, dsets[v].row_id) or not np.array_equal(ref.y, dsets[v].y):
            log.error("variant %s is not aligned with %s", v, variants[0])
            return 1
    y, groups, hn = ref.y, ref.groups, ref.hard_negative
    classes = ref.classes
    folds = make_folds(y, groups, n_splits, seed)
    from sentinel.dataset import load_cache

    fp = fingerprint(cfg, {v: str(load_cache(cfg, v)["cache_key"]) for v in variants}, folds)
    configs = expand_grid(cfg)
    Xs = {}
    for v in variants:
        for drop in m3["grid"]["drop_level_features"]:
            Xs[f"{v}|{int(drop)}"] = select_columns(dsets[v], cfg, bool(drop))
    print(f"{len(y)} windows, {len(set(groups))} source groups ({', '.join(f'{c}: {int(np.sum(y == c))}' for c in classes)}), "
          f"{n_splits} grouped folds, {len(configs)} configurations, fingerprint {fp}")

    # ---- 1. grid (resumable)
    path = tables / f"m3_grid_folds{tag}.jsonl"
    if args.force and path.exists():
        path.unlink()
    done = load_rows(path, fp)
    todo = missing_by_block_fold(done, configs, n_splits)
    n_cells = sum(len(v) for v in todo.values())
    print(f"grid: {len(done)} stored cells, {n_cells} to compute in {len(todo)} tasks")
    t0 = time.perf_counter()
    if todo:
        keys = list(todo)
        for bk in dict.fromkeys(k[0] for k in keys):  # one block at a time so progress is saved incrementally
            tasks = [k for k in keys if k[0] == bk]
            res = Parallel(n_jobs=args.n_jobs)(
                delayed(fold_worker)(Xs[bk], y, hn, folds[f][0], folds[f][1], todo[(bk, f)], cw, seed) for (_, f) in tasks)
            rows = []
            for (_, f), rs in zip(tasks, res):
                for r in rs:
                    rows.append({**r, "fold": f})
            append_rows(path, rows, fp)
            print(f"  block {bk:14s} {len(rows):4d} cells  ({time.perf_counter() - t0:.0f} s elapsed)")
    rows_df = load_rows(path, fp)
    wanted = {cfg_id(c) for c in configs}
    rows_df["cid"] = [cfg_id({"variant": r.variant, "drop_level": bool(r.drop_level), "pca": None if pd.isna(r.pca) else int(r.pca),
                              "kernel": r.kernel, "C": float(r.C)}) for r in rows_df.itertuples()]
    rows_df = rows_df[rows_df["cid"].isin(wanted)]
    summary = summarize_grid(rows_df.drop(columns=["cid"]))
    assert len(summary) == len(configs) and summary["n_folds"].eq(n_splits).all(), "grid incomplete"
    summary.to_csv(tables / f"m3_grid_summary{tag}.csv", index=False)
    effects = factor_effects(summary)
    effects.to_csv(tables / f"m3_factor_effects{tag}.csv", index=False)
    plot_ablation(summary, figs / f"m3_ablation{tag}.png")
    best = summary.iloc[0]
    d = m3["default"]
    dm = summary[(summary.variant == d["variant"]) & (summary.drop_level == d["drop_level_features"]) &
                 (summary.pca.astype("Int64") == (pd.NA if d["pca"] is None else d["pca"])) &
                 (summary.kernel == d["svm"]["kernel"]) & (summary.C == float(d["svm"]["C"]))] if d["pca"] is not None else summary.iloc[0:0]

    # ---- 3. nested selection
    nested = None
    if not args.skip_nested:
        import joblib

        npath = tables / f"m3_nested_cache{tag}.joblib"
        import hashlib

        ids_digest = hashlib.sha256("\n".join(cfg_id(c) for c in configs).encode()).hexdigest()[:12]  # stable across processes
        nfp = f"{fp}|{len(configs)}|{m3['nested']['inner_folds']}|{ids_digest}"
        if npath.exists() and not args.force:
            stored = joblib.load(npath)
            if stored.get("fp") == nfp:
                nested = stored["outer"]
                print("nested: loaded from cache")
        if nested is None:
            t1 = time.perf_counter()
            print(f"nested: {n_splits} outer folds x {m3['nested']['inner_folds']} inner folds x {len(configs)} configurations ...")
            nested = Parallel(n_jobs=args.n_jobs)(
                delayed(nested_outer_worker)(Xs, y, groups, hn, tr, te, configs, m3["nested"]["inner_folds"], cw, seed, i)
                for i, (tr, te) in enumerate(folds))
            joblib.dump({"fp": nfp, "outer": nested}, npath)
            print(f"  done in {time.perf_counter() - t1:.0f} s")

    # ---- 4. feature groups
    gm = feature_group_masks(ref.names, m3["feature_groups"])
    subsets = [("all", np.ones(len(ref.names), bool))]
    subsets += [(f"without:{g}", ~mk) for g, mk in gm.items()] + [(f"only:{g}", mk) for g, mk in gm.items()]
    gv = m3["group_study"]["variant"]
    if gv not in dsets:
        dsets[gv] = build_dataset(cfg, gv)
    Xg = dsets[gv].X
    gres = Parallel(n_jobs=args.n_jobs)(
        delayed(subset_worker)(Xg, y, hn, tr, te, subsets, m3["group_study"]["svm"], cw, seed) for tr, te in folds)
    gdf = pd.DataFrame([{**r, "fold": f} for f, rs in enumerate(gres) for r in rs])
    gsum = gdf.groupby("subset", sort=False).agg(n_features=("n_features", "first"),
                                                 **{f"{m}_mean": (m, "mean") for m in ("balanced_accuracy", "recall_aircraft", "specificity_plain", "specificity_hard_negative")},
                                                 **{f"{m}_std": (m, "std") for m in ("balanced_accuracy", "recall_aircraft")}).reset_index()
    piv = gdf.pivot(index="fold", columns="subset", values="balanced_accuracy")
    paired = piv.sub(piv["all"], axis=0)  # same fold, same pipeline: only the column subset differs
    gsum["ba_diff_vs_all_mean"] = gsum["subset"].map(paired.mean())
    gsum["ba_diff_vs_all_std"] = gsum["subset"].map(paired.std(ddof=1))
    gsum["folds_better_than_all"] = gsum["subset"].map((paired > 0).sum())
    gsum.to_csv(tables / f"m3_feature_groups{tag}.csv", index=False)
    full_ba = float(gsum[gsum["subset"] == "all"]["balanced_accuracy_mean"].iloc[0])
    plot_groups(gsum[gsum["subset"] != "all"], full_ba, figs / f"m3_feature_groups{tag}.png")

    # ---- results
    payload: dict = {
        "fingerprint": fp, "dataset": ref.info, "classes": classes, "n_windows": int(len(y)), "n_groups": int(len(set(groups))),
        "groups_per_class": {c: int(len(set(groups[y == c]))) for c in classes},
        "folds": f"StratifiedGroupKFold({n_splits}, shuffle, seed={seed}) on source group, identical for every configuration",
        "n_configs": len(configs), "class_weight": cw,
        "grid_top10": records(summary.head(10)),
        "grid_best": records(summary.head(1))[0],
        "grid_default_pipeline": records(dm) if len(dm) else None,
        "factor_effects": records(effects),
        "feature_groups": {"variant": gv, "svm": m3["group_study"]["svm"], "rows": records(gsum)},
        "note": "ESC-50 aircraft vs ambient only. Grid best is optimistic (selected on the data it is scored on); see nested."}
    if nested is not None:
        pred = np.empty(len(y), dtype=object)
        fold_rows = []
        for o in nested:
            pred[o["te"]] = o["pred"]
            fold_rows.append({"outer_fold": o["outer_fold"], "chosen": cfg_id(o["chosen"]), "inner_ba": o["inner_ba_chosen"],
                              "outer_ba": o["metrics"]["balanced_accuracy"], "outer_recall_aircraft": o["metrics"]["recall_aircraft"],
                              "outer_specificity_plain": o["metrics"]["specificity_plain"],
                              "outer_specificity_hard_negative": o["metrics"]["specificity_hard_negative"]})
        pred = pred.astype(str)
        agg = aggregate([o["metrics"] for o in nested])
        ci = group_bootstrap_ci(y, pred, groups, classes, m3["bootstrap"]["n"], seed)
        fa = false_alarm_table(ref, pred)
        pd.DataFrame(fold_rows).to_csv(tables / f"m3_nested_folds{tag}.csv", index=False)
        fa.to_csv(tables / f"m3_false_alarm_by_category_nested{tag}.csv", index=False)
        pd.DataFrame({"row_id": ref.row_id, "group": groups, "category": ref.category, "hard_negative": hn, "label": y, "pred": pred}
                     ).to_csv(tables / f"m3_oof_nested{tag}.csv", index=False)
        plot_confusion(y, pred, classes, figs / f"m3_confusion_nested{tag}.png", "nested selection, pooled out-of-fold")
        counts: dict[str, int] = {}
        for r in fold_rows:
            counts[r["chosen"]] = counts.get(r["chosen"], 0) + 1
        payload["nested"] = {
            "inner_folds": m3["nested"]["inner_folds"], "outer_folds": fold_rows, "fold_mean_std": agg,
            "pooled_bootstrap_ci95": ci, "chosen_counts": counts,
            "pooled_recall": {c: float(np.mean(pred[y == c] == c)) for c in classes},
            "pooled_balanced_accuracy": float(np.mean([np.mean(pred[y == c] == c) for c in classes])),
            "optimism_gap_best_grid_minus_nested": float(best["balanced_accuracy_mean"] - agg["balanced_accuracy_mean"])}
    payload["wall_seconds"] = round(time.perf_counter() - t_start, 1)
    update_results(cfg, f"ablation{tag}", payload)

    # ---- console report
    pd.set_option("display.width", 220, "display.max_columns", 30)
    show = ["variant", "drop_level", "pca", "kernel", "C", "balanced_accuracy_mean", "balanced_accuracy_std", "recall_aircraft_mean",
            "specificity_plain_mean", "specificity_hard_negative_mean", "precision_aircraft_mean", "auc_aircraft_mean"]
    print("\ntop 10 configurations by grouped-CV balanced accuracy (optimistic: picked on the data it is scored on)")
    print(summary.head(10)[show].round(3).to_string(index=False))
    if len(dm):
        print("\ndefault pipeline in the grid:")
        print(dm[show].round(3).to_string(index=False))
    print("\npaired factor effects on balanced accuracy (same configuration, one factor changed)")
    print(effects.round(4).to_string(index=False))
    print("\nfeature groups (variant %s, no PCA, rbf C=1); diff = paired per-fold difference to all features" % gv)
    print(gsum[["subset", "n_features", "balanced_accuracy_mean", "balanced_accuracy_std", "ba_diff_vs_all_mean", "ba_diff_vs_all_std",
                "folds_better_than_all", "recall_aircraft_mean", "specificity_plain_mean", "specificity_hard_negative_mean"]].round(3).to_string(index=False))
    if nested is not None:
        n = payload["nested"]
        print("\nnested selection, outer folds:")
        print(pd.DataFrame(fold_rows).round(3).to_string(index=False))
        a = n["fold_mean_std"]
        print(f"\nNESTED balanced accuracy {a['balanced_accuracy_mean']:.3f} +/- {a['balanced_accuracy_std']:.3f}; pooled "
              f"{n['pooled_balanced_accuracy']:.3f}, 95% group-bootstrap CI {n['pooled_bootstrap_ci95']['balanced_accuracy_lo']:.3f} to "
              f"{n['pooled_bootstrap_ci95']['balanced_accuracy_hi']:.3f}")
        print(f"  aircraft recall {a['recall_aircraft_mean']:.3f}, precision {a['precision_aircraft_mean']:.3f}, ambient specificity "
              f"plain {a['specificity_plain_mean']:.3f} / hard negatives {a['specificity_hard_negative_mean']:.3f}")
        print(f"  best grid cell minus nested estimate (selection optimism): {n['optimism_gap_best_grid_minus_nested']:+.3f}")
        print("  highest false-alarm categories:")
        print(fa.head(6).round(3).to_string(index=False))
    print(f"\nwrote tables m3_*{tag}.csv and figures m3_*{tag}.png  ({payload['wall_seconds']} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
