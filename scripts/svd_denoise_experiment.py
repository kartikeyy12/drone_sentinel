#!/usr/bin/env python
"""M2 SVD denoising experiment.

Part A (controlled): synthetic rotor-like and burst signals in white noise and in real ESC-50 ambient noise at fixed SNRs,
scored against the known clean magnitude spectrogram for specsub, Wiener, SVD (energy share sweep) and SVD (elbow).
Part B (fixed rank): magnitude SNR gain versus the rank kept, with the singular value spectra.
Part C (descriptive, no ground truth): rank kept, retained power and the effect of centring on real ESC-50 clips.

Writes results/tables/svd_*.csv, results/figures/svd_gain_vs_snr.png, svd_rank_sweep.png and results.json `svd_experiment`.
The synthetic signals are not drone recordings; see docs/m2_notes.md for what the numbers do and do not show.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sentinel import plotstyle  # noqa: E402
from sentinel.config import PROJECT_ROOT, load_config, project_path, set_seeds, setup_logging, update_results  # noqa: E402
from sentinel.preprocess import bandpass_offline, bandpass_sos, load_audio, preprocess_clip  # noqa: E402
from sentinel.svd_experiments import build_cases, mix_at_snr, rank_curve, real_clip_rank_stats, run_case  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

log = logging.getLogger("svd_denoise_experiment")
PLOT_METHODS = ["specsub", "wiener", "svd_elbow", "svd_energy_0.90", "svd_energy_0.95", "svd_energy_0.99"]
LABELS = {"specsub": "spectral subtraction", "wiener": "Wiener", "svd_elbow": "SVD elbow",
          "svd_energy_0.90": "SVD 90 % energy", "svd_energy_0.95": "SVD 95 % energy (default)",
          "svd_energy_0.99": "SVD 99 % energy"}
SIGNAL_TITLES = {"rotor": "steady rotor-like harmonics (synthetic)", "bursts": "impulsive bursts (synthetic)"}


def plot_gain(summary: pd.DataFrame, out: Path) -> None:
    plotstyle.apply()
    signals, kinds = ["rotor", "bursts"], ["white", "ambient"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, constrained_layout=True)
    handles = {}
    for r, sig in enumerate(signals):
        for c, kind in enumerate(kinds):
            ax = axes[r, c]
            for i, m in enumerate(PLOT_METHODS):
                d = summary[(summary.signal == sig) & (summary.noise_kind == kind) & (summary.method == m)].sort_values("snr_db")
                if d.empty:
                    continue
                err = d["mag_snr_gain_db_std"].fillna(0.0).to_numpy()
                h = ax.errorbar(d["snr_db"], d["mag_snr_gain_db_mean"], yerr=err if kind == "ambient" else None,
                                capsize=2, elinewidth=0.8, label=LABELS[m], **plotstyle.style(i))
                handles[m] = h
            ax.axhline(0, color=plotstyle.MUTED, lw=0.8)
            ax.set_title(f"{SIGNAL_TITLES[sig]}\nin {'white noise' if kind == 'white' else 'real ESC-50 ambient noise (mean, bars = std over clips)'}")
            if r == 1:
                ax.set_xlabel("input SNR after band-pass (dB)")
            if c == 0:
                ax.set_ylabel("magnitude SNR gain (dB)")
    fig.legend([handles[m] for m in PLOT_METHODS if m in handles], [LABELS[m] for m in PLOT_METHODS if m in handles],
               loc="outside lower center", ncol=3)
    plotstyle.save(fig, out)


def plot_rank(curves: dict, out: Path, ks_marks: dict) -> None:
    plotstyle.apply()
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    for r, sig in enumerate(["rotor", "bursts"]):
        ax_sv, ax_g = axes[r]
        w = curves[(sig, "white")]
        n = min(len(w["sv_clean"]), 80)
        ax_sv.semilogy(range(1, n + 1), w["sv_clean"][:n], label="clean", **plotstyle.style(0))
        ax_sv.semilogy(range(1, n + 1), w["sv_noisy"][:n], label="noisy (white, 0 dB)", **plotstyle.style(1))
        ax_sv.set_title(f"{SIGNAL_TITLES[sig]}\nsingular values of one 2 s |STFT| block")
        ax_sv.set_xlabel("component index")
        ax_sv.set_ylabel("singular value")
        ax_sv.legend()
        for i, kind in enumerate(["white", "ambient"]):
            cv = curves.get((sig, kind))
            if cv is None:
                continue
            ax_g.plot(cv["ks"], cv["mag_snr_gain_db"], label=f"{kind} noise", **plotstyle.style(i))
        ax_g.axhline(0, color=plotstyle.MUTED, lw=0.8)
        ax_g.set_xscale("log", base=2)
        ax_g.set_xlabel("rank kept (per 2 s block)")
        ax_g.set_ylabel("magnitude SNR gain (dB)")
        ax_g.set_title("gain versus fixed rank (input SNR 0 dB)")
        for label, k in ks_marks.get(sig, {}).items():
            ax_g.axvline(k, color=plotstyle.MUTED, lw=0.8, ls=":")
            ax_g.text(k, ax_g.get_ylim()[1], f" {label}", fontsize=7, color=plotstyle.MUTED, va="top", rotation=90)
        ax_g.legend(loc="lower right")
    plotstyle.save(fig, out)


def sample_real_clips(cfg: dict, man: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    n = cfg["m2"]["real_clips"]["per_group"]
    groups = {"aircraft": man[man["label"] == "aircraft"],
              "ambient": man[(man["label"] == "ambient") & (~man["hard_negative"])],
              "hard_negative": man[man["hard_negative"]]}
    picks = []
    for g, pool in groups.items():
        if pool.empty:
            print(f"no {g} clips in the manifest, skipping")
            continue
        idx = rng.choice(len(pool), size=min(n, len(pool)), replace=False)
        picks.append(pool.iloc[np.sort(idx)].assign(group=g))
    return pd.concat(picks, ignore_index=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--n-jobs", type=int, default=-1)
    args = ap.parse_args()
    setup_logging()
    cfg = load_config(args.config)
    set_seeds(cfg["seed"])
    man_path = project_path(cfg, "manifest")
    if not man_path.exists():
        log.error("%s not found. Run: python scripts/build_manifest.py", man_path)
        return 1
    man = pd.read_csv(man_path)
    rng = np.random.default_rng(cfg["seed"])
    tables, figs = project_path(cfg, "tables"), project_path(cfg, "figures")
    tables.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    # ---- Part A: controlled sweep
    cases = build_cases(cfg, man, rng)
    noise_ids = sorted({(c["noise_kind"], c["noise_id"]) for c in cases})
    print(f"controlled sweep: {len(cases)} cases, {len(noise_ids)} noise sources "
          f"({sum(k == 'ambient' for k, _ in noise_ids)} real ambient clips), n_jobs={args.n_jobs}")
    rows = [r for rs in Parallel(n_jobs=args.n_jobs, batch_size=1)(delayed(run_case)(c, cfg) for c in cases) for r in rs]
    df = pd.DataFrame(rows)
    df.to_csv(tables / "svd_synthetic_cases.csv", index=False)
    summary = (df.groupby(["signal", "noise_kind", "snr_db", "method"])
                 .agg(n=("mag_snr_gain_db", "size"), mag_snr_gain_db_mean=("mag_snr_gain_db", "mean"),
                      mag_snr_gain_db_std=("mag_snr_gain_db", "std"), time_snr_gain_db_mean=("time_snr_gain_db", "mean"),
                      time_snr_gain_db_std=("time_snr_gain_db", "std"), k_mean=("k_mean", "mean"))
                 .reset_index())
    order = {m: i for i, m in enumerate(df["method"].unique())}
    summary = summary.sort_values(["signal", "noise_kind", "snr_db", "method"], key=lambda s: s.map(order) if s.name == "method" else s)
    summary.to_csv(tables / "svd_synthetic_summary.csv", index=False)
    plot_gain(summary, figs / "svd_gain_vs_snr.png")

    # ---- Part B: fixed-rank curves at one SNR, white noise and the first ambient clip
    sw = cfg["m2"]["sweep"]
    snr0 = float(sw["rank_curve_snr_db"])
    sr = cfg["audio"]["sr"]
    sos = bandpass_sos(sr, cfg["preprocess"]["bandpass"]["low_hz"], cfg["preprocess"]["bandpass"]["high_hz"],
                       cfg["preprocess"]["bandpass"]["order"])
    curves: dict = {}
    ks_marks: dict = {}
    for sig in ("rotor", "bursts"):
        for kind in ("white", "ambient"):
            sel = [c for c in cases if c["signal"] == sig and c["noise_kind"] == kind and c["snr_db"] == snr0
                   and c["realization"] == 0]
            if not sel:
                continue
            c = sel[0]
            clean = bandpass_offline(c["clean"].astype(np.float32), sos).astype(np.float64)
            noise = bandpass_offline(c["noise"].astype(np.float32), sos).astype(np.float64)
            noisy, _ = mix_at_snr(clean, noise, snr0)
            curves[(sig, kind)] = {**rank_curve(clean, noisy, cfg, sw["fixed_rank"]), "noise_id": c["noise_id"]}
        ks = {}
        w = df[(df.signal == sig) & (df.noise_kind == "white") & (df.snr_db == snr0)]
        for m in ("svd_elbow", "svd_energy_0.90", "svd_energy_0.99"):
            if m in set(w["method"]):
                ks[{"svd_elbow": "elbow", "svd_energy_0.90": "90 %", "svd_energy_0.99": "99 %"}[m]] = float(w[w.method == m]["k_mean"].mean())
        ks_marks[sig] = ks
    plot_rank(curves, figs / "svd_rank_sweep.png", ks_marks)

    # ---- Part C: real clips, no ground truth
    pick = sample_real_clips(cfg, man, np.random.default_rng([cfg["seed"], 2]))  # own stream: independent of the sweep
    real_rows, centering_rows = [], []
    for _, row in pick.iterrows():
        p = PROJECT_ROOT / row["path"]
        x = preprocess_clip(load_audio(p, sr), sr, cfg)
        st = real_clip_rank_stats(x, cfg)
        for m, v in st.items():
            if m == "centering":
                centering_rows.append({"group": row["group"], "path": row["path"], **v})
            else:
                real_rows.append({"group": row["group"], "path": row["path"], "method": m, **v})
    real = pd.DataFrame(real_rows)
    real_sum = real.groupby(["group", "method"]).agg(n_clips=("path", "nunique"), k_mean=("k_mean", "mean"),
                                                     power_change_db_mean=("power_change_db", "mean"),
                                                     power_change_db_std=("power_change_db", "std")).reset_index()
    real_sum.to_csv(tables / "svd_real_clip_sweep.csv", index=False)
    cen = pd.DataFrame(centering_rows)
    cen_sum = cen.drop(columns="path").groupby("group").mean().reset_index()
    cen_sum.to_csv(tables / "svd_centering_real_clips.csv", index=False)

    # ---- results.json
    payload = {
        "config": cfg["m2"], "seed": cfg["seed"], "n_cases": len(cases), "noise_sources": [list(k) for k in noise_ids],
        "synthetic_summary": summary.round(4).to_dict(orient="records"),
        "rank_curves": {f"{s}|{k}": v for (s, k), v in curves.items()},
        "rank_curve_marks_white_noise": ks_marks,
        "real_clips": {"files": pick[["group", "path"]].to_dict(orient="records"),
                       "summary": real_sum.round(4).to_dict(orient="records"),
                       "centering": cen_sum.round(4).to_dict(orient="records"),
                       "note": "no clean reference exists for real clips; these are rank and power statistics, not quality scores"},
        "wall_seconds": round(time.perf_counter() - t0, 1),
    }
    update_results(cfg, "svd_experiment", payload)

    # ---- console report
    pd.set_option("display.width", 200, "display.max_columns", 30)
    print("\nmagnitude SNR gain in dB (mean over clips/realizations), positive is better")
    for sig in ("rotor", "bursts"):
        for kind in ("white", "ambient"):
            s = summary[(summary.signal == sig) & (summary.noise_kind == kind)]
            pv = s.pivot(index="method", columns="snr_db", values="mag_snr_gain_db_mean").reindex(df["method"].unique())
            print(f"\n{sig} in {kind} noise (n={int(s['n'].iloc[0])} per cell), columns = input SNR dB")
            print(pv.round(2).to_string())
    print("\nreal clips, mean rank kept (at most min(257, frames per block) = 250 available) and power change in dB")
    print(real_sum.pivot(index="method", columns="group", values="k_mean").round(1).to_string())
    print(real_sum.pivot(index="method", columns="group", values="power_change_db_mean").round(2).to_string())
    print("\ncentring effect on real clips (mean over blocks and clips)")
    print(cen_sum.round(3).to_string(index=False))
    print(f"\nwrote {tables}/svd_*.csv, {figs}/svd_gain_vs_snr.png, {figs}/svd_rank_sweep.png  ({payload['wall_seconds']} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
