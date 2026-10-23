"""M4 figures: robustness curves, calibration reliability and the operating curve. Each series has its own colour, line
style and marker, so nothing depends on colour alone."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import plotstyle


def _xlabels(snrs: list[float]) -> list[str]:
    return ["clean"] + [f"{s:+g}" for s in snrs]


def _x(df_kind: pd.DataFrame, snrs: list[float], col: str) -> np.ndarray:
    """Values in x order (clean, then the SNRs from mild to severe)."""
    out = []
    for s in [None] + snrs:
        r = df_kind[df_kind["snr_db"].isna()] if s is None else df_kind[df_kind["snr_db"] == s]
        out.append(float(r.iloc[0][col]) if len(r) else np.nan)
    return np.array(out)


def plot_robustness_ba(recs: pd.DataFrame, aug: pd.DataFrame | None, kinds: list[str], snrs: list[float],
                       variants: list[str], path: Path, pos: str) -> Path:
    """Pooled out-of-fold balanced accuracy with group-bootstrap CI, one panel per noise kind, one line per variant
    (trained on clean audio). Dashed black: trained with the other noise kinds, tested on this one."""
    import matplotlib.pyplot as plt

    plotstyle.apply()
    snrs = sorted(snrs, reverse=True)
    fig, axes = plt.subplots(1, len(kinds), figsize=(3.6 * len(kinds), 3.7), sharey=True, constrained_layout=True)
    axes = np.atleast_1d(axes)
    xs = np.arange(len(snrs) + 1)
    for ax, kind in zip(axes, kinds):
        k = recs[(recs["kind"] == kind) | (recs["kind"] == "clean")]
        series = [(v, k[k["variant"] == v], plotstyle.style(i)) for i, v in enumerate(variants)]
        if aug is not None and len(aug):
            a = aug[(aug["held_out_kind"] == kind)]
            series.append((f"{a.iloc[0]['variant']}, trained with other noise kinds" if len(a) else "", a,
                           {"color": plotstyle.INK, "linestyle": (0, (4, 2)), "marker": "x", "markeredgewidth": 1.4}))
        for j, (name, d, st) in enumerate(series):
            if not len(d) or not name:
                continue
            y, lo, hi = _x(d, snrs, "ba_pooled"), _x(d, snrs, "ba_lo"), _x(d, snrs, "ba_hi")
            off = (j - (len(series) - 1) / 2) * 0.07
            ax.errorbar(xs + off, y, yerr=[y - lo, hi - y], capsize=2, elinewidth=0.9, label=name, **st)
        ax.axhline(0.5, color=plotstyle.MUTED, linewidth=0.8, linestyle=":")
        ax.text(-0.45, 0.497, "chance", fontsize=7, color=plotstyle.MUTED, ha="left", va="top")
        ax.set_xticks(xs, _xlabels(snrs))
        ax.set_title(f"{kind} noise")
        ax.set_xlabel("SNR against the median aircraft clip (dB)")
    axes[0].set_ylabel("balanced accuracy (pooled, 95% CI)")
    axes[0].set_ylim(0.4, 0.95)
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="outside lower center", ncol=min(4, len(lab)))
    return plotstyle.save(fig, path)


def plot_robustness_components(recs: pd.DataFrame, variant: str, kinds: list[str], snrs: list[float], path: Path,
                               pos: str) -> Path:
    """What moves when noise is added, for one variant: target recall, ambient specificity (plain and hard negatives),
    AUC (threshold free) and specificity on the noise-only windows. All are 'higher is better'."""
    import matplotlib.pyplot as plt

    plotstyle.apply()
    snrs = sorted(snrs, reverse=True)
    d = recs[recs["variant"] == variant].copy()
    d["noise_only_specificity"] = 1.0 - d["noise_only_far"]
    cols = [(f"recall_{pos}", f"{pos} recall"), ("specificity_plain", "ambient specificity"),
            ("specificity_hard_negative", "hard-negative specificity"), (f"auc_{pos}", f"{pos} AUC"),
            ("noise_only_specificity", "noise-only windows, correct")]
    fig, axes = plt.subplots(1, len(kinds), figsize=(3.6 * len(kinds), 3.7), sharey=True, constrained_layout=True)
    axes = np.atleast_1d(axes)
    xs = np.arange(len(snrs) + 1)
    for ax, kind in zip(axes, kinds):
        k = d[(d["kind"] == kind) | (d["kind"] == "clean")]
        for i, (c, name) in enumerate(cols):
            ax.plot(xs, _x(k, snrs, c), label=name, **plotstyle.style(i))
        ax.set_xticks(xs, _xlabels(snrs))
        ax.set_title(f"{kind} noise")
        ax.set_xlabel("SNR against the median aircraft clip (dB)")
    axes[0].set_ylabel(f"{variant}, trained on clean audio")
    axes[0].set_ylim(0.0, 1.02)
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="outside lower center", ncol=3)
    return plotstyle.save(fig, path)


def plot_calibration(bins: pd.DataFrame, path: Path, title: str) -> Path:
    """Reliability: mean predicted probability against observed frequency, equal-count bins, log-log because the
    positive class is rare. Marker area is not used; the bin counts are in the CSV."""
    import matplotlib.pyplot as plt

    plotstyle.apply()
    fig, ax = plt.subplots(figsize=(4.4, 4.2), constrained_layout=True)
    lo = max(1e-3, float(min(bins["mean_predicted"].min(), bins["observed"].replace(0, np.nan).min())) * 0.7)
    ax.plot([lo, 1], [lo, 1], color=plotstyle.MUTED, linestyle=":", linewidth=1.0, label="perfect")
    for i, (name, g) in enumerate(bins.groupby("kind", sort=False)):
        g = g[g["observed"] > 0]
        ax.plot(g["mean_predicted"], g["observed"], label=name, **plotstyle.style(i))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("mean predicted probability")
    ax.set_ylabel("observed frequency of the target class")
    ax.set_title(title, fontsize=9)
    ax.legend(loc="upper left")
    return plotstyle.save(fig, path)


def plot_operating_curve(curve: pd.DataFrame, points: pd.DataFrame, path: Path, pos: str, title: str) -> Path:
    """Target recall against the window-level false-alarm rate on ambient windows, from pooled out-of-fold scores,
    with the cross-fitted operating points (threshold chosen on the other folds, applied to the held-out fold)."""
    import matplotlib.pyplot as plt

    plotstyle.apply()
    fig, ax = plt.subplots(figsize=(4.8, 4.2), constrained_layout=True)
    ax.plot(curve["far"], curve["recall"], label="pooled out-of-fold ROC", **{**plotstyle.style(0), "marker": None})
    st = {k: v for k, v in plotstyle.style(1).items() if k != "linestyle"}
    ax.errorbar(points["far_mean"], points["recall_mean"], xerr=points["far_std"], yerr=points["recall_std"], fmt="none",
                ecolor=plotstyle.SERIES[1], elinewidth=0.9, capsize=2)
    ax.plot(points["far_mean"], points["recall_mean"], linestyle="none", label="cross-fitted operating points (mean, bars = fold std)", **st)
    for r in points.itertuples():
        ax.annotate(f"target {r.target_far:.0%}", (r.far_mean, r.recall_mean), textcoords="offset points", xytext=(6, -12),
                    fontsize=7, color=plotstyle.MUTED)
    ax.set_xlabel("false-alarm rate on ambient windows")
    ax.set_ylabel(f"{pos} recall")
    ax.set_xlim(0, max(0.35, float(points["far_mean"].max()) * 1.1))
    ax.set_ylim(0, 1.0)
    ax.set_title(title, fontsize=9)
    ax.legend(loc="lower right")
    return plotstyle.save(fig, path)
