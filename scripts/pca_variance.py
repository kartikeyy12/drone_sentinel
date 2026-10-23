#!/usr/bin/env python
"""M2 PCA variance analysis of the cached window features, one SVDPCA fit per denoising variant.

Descriptive only. Features are z-scored with statistics from ALL windows and the PCA sees no labels, so these numbers
describe the feature matrix and must not be used to pick a model: in M3 the scaler and PCA are fitted inside each
training fold. Class labels are used only to colour the scatter plot and to name the rows of the level-feature check.

Writes results/tables/pca_*.csv, results/figures/pca_variance.png, pca_scatter.png and results.json `pca_variance`.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sentinel import plotstyle  # noqa: E402
from sentinel.config import PROJECT_ROOT, load_config, project_path, set_seeds, setup_logging, update_results  # noqa: E402
from sentinel.svdpca import SVDPCA  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

log = logging.getLogger("pca_variance")
SILENT_RMS = 1e-4  # mean frame RMS below this (about -80 dBFS) counts as a near-silent window
VARIANT_LABELS = {"none": "band-pass only", "specsub": "spectral subtraction", "wiener": "Wiener",
                  "svd_energy": "SVD energy", "svd_elbow": "SVD elbow"}


def zscore(X: np.ndarray) -> np.ndarray:
    sd = X.std(axis=0)
    return (X - X.mean(axis=0)) / np.where(sd > 0, sd, 1.0)


def components_for(ratio: np.ndarray, target: float) -> int:
    """Smallest k whose cumulative ratio reaches the target."""
    return int(np.searchsorted(np.cumsum(ratio), target - 1e-12) + 1)


def load_cache(cfg: dict, variant: str) -> dict | None:
    p = PROJECT_ROOT / cfg["features"]["cache_dir"] / f"features_{variant}.npz"
    if not p.exists():
        print(f"missing {p}, skipping {variant} (run: make features)")
        return None
    with np.load(p, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    setup_logging()
    cfg = load_config(args.config)
    set_seeds(cfg["seed"])
    pc = cfg["m2"]["pca"]
    targets = pc["variance_targets"]
    tables, figs = project_path(cfg, "tables"), project_path(cfg, "figures")
    tables.mkdir(parents=True, exist_ok=True)

    fits: dict[str, dict] = {}
    rows = []
    for v in cfg["variants"]:
        z = load_cache(cfg, v)
        if z is None:
            continue
        names = [str(n) for n in z["feature_names"]]
        level = np.array([any(n.startswith(p) for p in pc["level_feature_prefixes"]) for n in names])
        Z = zscore(z["X"].astype(np.float64))
        full = SVDPCA().fit(Z)
        nolevel = SVDPCA().fit(Z[:, ~level])
        silent = z["X"][:, names.index("rms_mean")] < SILENT_RMS  # same definition as scripts/audit_silence.py
        quiet_fit = SVDPCA().fit(zscore(z["X"][~silent].astype(np.float64)))
        fits[v] = {"z": z, "names": names, "level": level, "full": full, "nolevel": nolevel, "Z": Z, "silent": silent}
        row = {"variant": v, "n_windows": len(Z), "n_features": Z.shape[1], "n_level_features": int(level.sum()),
               "pc1_ratio": float(full.explained_variance_ratio_all_[0]),
               "pc1_to_3_ratio": float(full.explained_variance_ratio_all_[:3].sum())}
        row["n_near_silent_windows"] = int(silent.sum())
        row["pc1_ratio_without_near_silent"] = float(quiet_fit.explained_variance_ratio_all_[0])
        row["k95_without_near_silent"] = components_for(quiet_fit.explained_variance_ratio_all_, 0.95)
        for t in targets:
            row[f"k{int(round(t * 100))}"] = components_for(full.explained_variance_ratio_all_, t)
            row[f"k{int(round(t * 100))}_without_level_features"] = components_for(nolevel.explained_variance_ratio_all_, t)
        rows.append(row)
    if not rows:
        log.error("no feature caches found. Run: make features")
        return 1
    table = pd.DataFrame(rows)
    table.to_csv(tables / "pca_components_needed.csv", index=False)

    # ---- level-feature check on the band-pass-only variant (or the first available one)
    ref = "none" if "none" in fits else next(iter(fits))
    f = fits[ref]
    model = f["full"]
    scores = model.transform(f["Z"])
    names = f["names"]
    probes = [n for n in ("rms_mean", "mfcc00_mean", "mfcc00_std", "centroid_mean", "zcr_mean") if n in names]
    corr_rows = []
    for n in probes:
        col = f["z"]["X"][:, names.index(n)]
        for j in range(3):
            rho = float(spearmanr(scores[:, j], col).statistic)
            corr_rows.append({"variant": ref, "feature": n, "component": f"PC{j + 1}", "spearman": rho})
    corr = pd.DataFrame(corr_rows)
    corr.to_csv(tables / "pca_level_features.csv", index=False)
    loadings = []
    for j in range(3):
        top = np.argsort(-np.abs(model.components_[j]))[:6]
        loadings.append({"component": f"PC{j + 1}", "top_features": [[names[i], round(float(model.components_[j][i]), 3)] for i in top]})

    silent = f["silent"]
    left = scores[:, 0] < -15
    stack = {"reference_variant": ref, "n_windows_pc1_below_-15": int(left.sum()),
             "share_of_those_that_are_near_silent": float(silent[left].mean()) if left.any() else None,
             "n_near_silent_windows": int(silent.sum()),
             "share_of_near_silent_with_pc1_below_-15": float(left[silent].mean()) if silent.any() else None,
             "near_silent_by_label": {k: int(v) for k, v in pd.Series(f["z"]["label"].astype(str)[silent]).value_counts().items()}}

    # ---- figures
    plotstyle.apply()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    n_show = min(len(fits[ref]["full"].explained_variance_ratio_all_), 100)
    for i, (v, fv) in enumerate(fits.items()):
        cum = np.cumsum(fv["full"].explained_variance_ratio_all_)[:n_show]
        a1.plot(np.arange(1, n_show + 1), cum, label=VARIANT_LABELS[v], markevery=10, **plotstyle.style(i))
    for t in targets:
        a1.axhline(t, color=plotstyle.MUTED, lw=0.7, ls=":")
        a1.text(n_show, t, f"{t:.0%} ", ha="right", va="bottom", fontsize=7, color=plotstyle.MUTED)
    a1.set_xlabel("number of components")
    a1.set_ylabel("cumulative explained variance")
    a1.set_title("z-scored window features, by denoising variant")
    a1.legend(loc="lower right")
    m = pc["scree_components"]
    r_all = f["full"].explained_variance_ratio_all_[:m]
    r_nl = f["nolevel"].explained_variance_ratio_all_[:m]
    a2.plot(np.arange(1, len(r_all) + 1), r_all, label=f"all {int(table.n_features.iloc[0])} features", **plotstyle.style(0))
    a2.plot(np.arange(1, len(r_nl) + 1), r_nl, label=f"without {int(f['level'].sum())} level features", **plotstyle.style(1))
    a2.set_yscale("log")
    a2.set_xlabel("component index")
    a2.set_ylabel("explained variance ratio")
    a2.set_title(f"scree, {VARIANT_LABELS[ref]}")
    a2.legend()
    plotstyle.save(fig, figs / "pca_variance.png")

    fig, ax = plt.subplots(figsize=(6.4, 5.2), constrained_layout=True)
    lab = f["z"]["label"].astype(str)
    hn = f["z"]["hard_negative"].astype(bool)
    groups = [("aircraft", (lab == "aircraft"), 0), ("ambient", (lab == "ambient") & ~hn, 1), ("ambient, hard negative", hn, 2)]
    for name, mask, i in sorted(groups, key=lambda g: -int(g[1].sum())):  # largest first, aircraft ends on top
        st = plotstyle.style(i)
        ax.scatter(scores[mask, 0], scores[mask, 1], s=7, alpha=0.4, color=st["color"], marker=st["marker"],
                   linewidths=0, label=f"{name} ({int(mask.sum())})")
    ax.set_xlabel(f"PC1 ({model.explained_variance_ratio_all_[0]:.1%} of variance)")
    ax.set_ylabel(f"PC2 ({model.explained_variance_ratio_all_[1]:.1%})")
    ax.set_title(f"first two principal components, {VARIANT_LABELS[ref]}\n(descriptive: scaler and PCA fitted on all windows)")
    leg = ax.legend(markerscale=2.5)
    for h in leg.legend_handles:
        h.set_alpha(1)
    plotstyle.save(fig, figs / "pca_scatter.png")

    update_results(cfg, "pca_variance", {
        "reference_variant": ref, "variance_targets": targets,
        "level_feature_prefixes": pc["level_feature_prefixes"],
        "per_variant": table.round(5).to_dict(orient="records"),
        "level_feature_spearman": corr.round(4).to_dict(orient="records"),
        "top_loadings": loadings, "near_silent_check": stack,
        "note": "descriptive only: z-scoring and PCA use all windows, no labels; M3 fits both inside training folds",
    })

    pd.set_option("display.width", 220, "display.max_columns", 30)
    print("components needed for 90 / 95 / 99 % of the variance (z-scored features)\n")
    print(table.to_string(index=False))
    print(f"\nSpearman correlation of {ref} component scores with level-related features")
    print(corr.pivot(index="feature", columns="component", values="spearman").round(3).to_string())
    print("\nnear-silent windows (rms_mean < 1e-4) versus the PC1 < -15 stack:", stack)
    for ld in loadings:
        print(f"{ld['component']} top loadings: " + ", ".join(f"{n} {w:+.2f}" for n, w in ld["top_features"]))
    print(f"\nwrote {tables}/pca_*.csv, {figs}/pca_variance.png, {figs}/pca_scatter.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
