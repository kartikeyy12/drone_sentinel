"""M7: range model. Reads results.json, writes results.json -> range, tables and figures; then runs the bench script if bench files exist.

    make range

snr_min_db is the lowest SNR from which the recall stays at or above m7.recall_target on the m7.primary_curve. If that is never
reached the output says so and the range is shown only as a sensitivity table for hypothetical required SNRs.
"""
from __future__ import annotations

import json
import logging
import sys

import pandas as pd

from sentinel import plotstyle as ps
from sentinel.config import load_config, project_path, setup_logging, update_results
from sentinel.range_model import analyse


def main() -> int:
    setup_logging(logging.WARNING)
    cfg = load_config()
    res = json.loads(project_path(cfg, "results_json").read_text(encoding="utf-8"))
    a = analyse(res, cfg)
    tables, figs = project_path(cfg, "tables"), project_path(cfg, "figures")
    tables.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(a["sensitivity"]).to_csv(tables / "m7_range_sensitivity.csv", index=False)
    cur = [{"curve": k, "unit": c["unit"], "snr_db": s, "recall": r} for k, c in a["curves"].items() for s, r in zip(c["snr_db"], c["recall"])]
    pd.DataFrame(cur).to_csv(tables / "m7_recall_curves.csv", index=False)
    ps.apply()
    # sensitivity plot: one panel per hypothetical required SNR, one line per ambient level
    sn = a["sensitivity_hypothetical_snr_min_db"]
    df = pd.DataFrame(a["sensitivity"])
    fig, axs = ps.plt.subplots(1, len(sn), figsize=(3.6 * len(sn), 3.4), sharey=True)
    for ax, s in zip(ps.plt.np.atleast_1d(axs), sn):
        for i, la in enumerate(cfg["m7"]["sensitivity"]["ambient_levels_db"]):
            d = df[(df.snr_min_db == s) & (df.ambient_level_db == la)].dropna(subset=["range_m"])
            ax.plot(d.source_level_db, d.range_m, label=f"ambient {la} dB", **ps.style(i))
        ax.set_yscale("log")
        ax.set_xlabel("source level at 1 m (dB)")
        ax.set_title(f"required SNR {s:+g} dB")
    ps.plt.np.atleast_1d(axs)[0].set_ylabel("range (m, log scale)")
    ps.plt.np.atleast_1d(axs)[0].legend()
    fig.suptitle("Model estimate of detection range (free field, hypothetical required SNR)", fontsize=10)
    fig.tight_layout()
    ps.save(fig, figs / "m7_range_sensitivity.png")
    # recall versus SNR curves with the target
    fig, ax = ps.plt.subplots(figsize=(6.4, 3.8))
    for i, (k, c) in enumerate(a["curves"].items()):
        ax.plot(c["snr_db"], c["recall"], label=k, **ps.style(i))
    ax.axhline(a["recall_target"], color=ps.MUTED, linestyle=":", linewidth=1.2)
    ax.text(ax.get_xlim()[0], a["recall_target"] + 0.01, f"target {a['recall_target']:g}", color=ps.MUTED, fontsize=8)
    ax.set_xlabel("SNR (dB), axis differs by curve")
    ax.set_ylabel("recall")
    ax.set_ylim(0, 1.02)
    ax.set_title("Recall versus SNR: where snr_min_db would come from")
    ax.legend(fontsize=7)
    fig.tight_layout()
    ps.save(fig, figs / "m7_recall_curves.png")
    payload = {k: v for k, v in a.items() if k != "sensitivity"}
    payload["sensitivity_rows"] = len(a["sensitivity"])
    payload["sensitivity_csv"] = "results/tables/m7_range_sensitivity.csv"
    update_results(cfg, "range", payload)
    print(f"primary curve {a['primary_curve']}, target recall {a['recall_target']:g}")
    if a["snr_min_reached"]:
        print(f"snr_min_db = {a['snr_min_db']:+.1f} dB")
    else:
        best = a["curve_summary"][a["primary_curve"]]
        print(f"snr_min_db: NOT REACHED (best recall {best['max_recall']:.3f} at {best['snr_at_max_recall_db']:+.0f} dB)")
    for m in a["range_estimate_missing_inputs"]:
        print("missing:", m)
    if a["range_estimate_m"] is not None:
        print(f"range estimate {a['range_estimate_m']:.1f} m (model estimate, not a measurement)")
    print("wrote results.json `range`, results/tables/m7_*.csv, results/figures/m7_*.png")
    from bench_distance import main as bench_main  # same folder
    return bench_main([])


if __name__ == "__main__":
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
    sys.exit(main())
