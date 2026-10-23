#!/usr/bin/env python
"""M5 `make stream-check`: does the live engine do what the offline chain did, and how fast and accurate is it?

  A  chunk invariance   the engine gives the same probabilities and events for any chunk size (real ESC-50 clips)
  B  causal vs offline  the engine filters causally (sosfilt from zero state), training used zero-phase filtering. Features
                        of every ESC-50 window are recomputed the live way; a model trained on zero-phase features is scored
                        on them with the M3 grouped folds (same folds, same pipeline). Also: a model trained on the causal
                        features, to see whether retraining would remove the gap.
  C  speed              real-time factor and per-window time of the engine on this machine
  D  SNR estimate       known mixtures (pink noise + an aircraft clip at a set in-band SNR) through the engine; per-window
                        estimate against the truth

Writes results.json `stream_check`, results/tables/m5_*.csv and results/figures/m5_*.png. ESC-50 only (aircraft versus
ambient); nothing here is a drone result.
"""
from __future__ import annotations

import argparse
import logging
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sentinel import plotstyle  # noqa: E402
from sentinel import robustness as rb  # noqa: E402
from sentinel import streamcheck as sc  # noqa: E402
from sentinel.bundle import load_model  # noqa: E402
from sentinel.config import PROJECT_ROOT, load_config, project_path, set_seeds, setup_logging, update_results  # noqa: E402
from sentinel.cv import make_folds  # noqa: E402
from sentinel.dataset import build_dataset, gated_ambient_windows  # noqa: E402
from sentinel.engine import run_offline  # noqa: E402
from sentinel.features import feature_names  # noqa: E402
from sentinel.preprocess import load_audio  # noqa: E402

log = logging.getLogger("check_streaming")


def usable_clips(cfg, manifest: pd.DataFrame, label: str) -> list[str]:
    m = manifest[(manifest["label"] == label) & ~manifest["dataset"].isin(cfg["splits"]["holdout_datasets"])]
    return [str(PROJECT_ROOT / p) for p in m["path"]]


def spread(items: list[str], n: int) -> list[str]:
    """n items spread evenly over the list (deterministic)."""
    idx = np.unique(np.linspace(0, len(items) - 1, n).round().astype(int))
    return [items[i] for i in idx]


# ---- A ---------------------------------------------------------------------------------------------------------------------

def chunk_invariance(cfg, model, clips: list[str]) -> dict:
    sizes = list(cfg["m5"]["check"]["chunk_sizes_s"])
    sr = cfg["audio"]["sr"]
    worst, same_events = 0.0, True
    for path in clips:
        x = load_audio(path, sr)
        ref_w, ref_r, _ = run_offline(model, cfg, x, chunk_s=sizes[0], t0_wall=0.0)
        for s in sizes[1:]:
            w, r, _ = run_offline(model, cfg, x, chunk_s=s, t0_wall=0.0)
            assert len(w) == len(ref_w), f"window count differs for chunk {s}"
            worst = max(worst, max(abs(a.p[t] - b.p[t]) for a, b in zip(w, ref_w) for t in a.p))
            same_events &= r == ref_r
    return {"n_clips": len(clips), "chunk_sizes_s": sizes, "max_abs_dp": worst, "events_identical": bool(same_events)}


# ---- B ---------------------------------------------------------------------------------------------------------------------

def flat(rec: dict) -> dict:
    out = {k: v for k, v in rec.items() if k != "ba_folds"}
    out.update({f"ba_fold{i}": b for i, b in enumerate(rec["ba_folds"])})
    return out


def causal_vs_offline(cfg, model, args) -> tuple[dict, pd.DataFrame, dict]:
    manifest = pd.read_csv(project_path(cfg, "manifest"))
    n_jobs = cfg["m5"]["check"]["n_jobs"] or cfg["features"]["n_jobs"]
    build = sc.build_causal(cfg, manifest, n_jobs=n_jobs, force=args.force)
    print(f"[B] causal feature cache: {'built in ' + str(build['wall_seconds']) + ' s' if build['built'] else 'cache hit'}", flush=True)
    seed, n_boot = cfg["seed"], int(cfg["m5"]["check"]["n_bootstrap"])
    ds, gated = build_dataset(cfg, "none"), gated_ambient_windows(cfg)
    folds = make_folds(ds.y, ds.groups, cfg["splits"]["cv_folds"], seed)
    feats = sc.CausalFeatures(cfg)
    Xc_all = feats.get("none", "causal", None)
    clean, causal = rb.CLEAN, ("causal", None)
    r1 = rb.evaluate_conditions(cfg, ds, gated, folds, "none", feats, [clean, causal], rb.clean_train(ds), n_boot, seed,
                                "trained_zero_phase", label="trained on zero-phase features")
    r2 = rb.evaluate_conditions(cfg, ds, gated, folds, "none", feats, [causal], lambda i, tr: Xc_all[ds.row_id[tr]], n_boot, seed,
                                "trained_causal", label="trained on causal features")
    recs = pd.DataFrame([flat(r) for r in r1 + r2])
    recs["tested_on"] = recs["kind"].map({"clean": "zero-phase features", "causal": "causal features"})
    pos = cfg["m4"]["positive_classes"][0]

    def row(tag, tested):
        return recs[(recs["tag"] == tag) & (recs["tested_on"] == tested)].iloc[0]
    a, b, c = row("trained_zero_phase", "zero-phase features"), row("trained_zero_phase", "causal features"), row("trained_causal", "causal features")
    fold_diff = np.array(b["ba_folds"] if "ba_folds" in b else [b[f"ba_fold{i}"] for i in range(len(folds))]) \
        - np.array([a[f"ba_fold{i}"] for i in range(len(folds))])

    # paired comparison with the saved model (trained on all windows, so it has seen these: descriptive only)
    rms = feature_names(cfg["features"]).index("rms_mean")
    Xz, Xc = ds.X, Xc_all[ds.row_id].astype(np.float64)
    pz, gz = model.score_features(Xz, Xz[:, rms])
    pc, gc = model.score_features(Xc, Xc[:, rms])
    thr = model.thresholds[pos]
    dp = np.abs(pz[:, 0] - pc[:, 0])
    ref = load_cache_rms(cfg)
    gate_flips = int(np.sum((ref[0] < model.silence_gate_rms) != (ref[1] < model.silence_gate_rms)))
    pair = {"n_windows": int(len(dp)), "mean_abs_dp": float(dp.mean()), "p95_abs_dp": float(np.percentile(dp, 95)),
            "max_abs_dp": float(dp.max()), "alert_decision_flips": int(np.sum((pz[:, 0] >= thr) != (pc[:, 0] >= thr))),
            "alert_flip_fraction": float(np.mean((pz[:, 0] >= thr) != (pc[:, 0] >= thr))),
            "gate_flips_all_rows": gate_flips, "threshold": float(thr),
            "note": "saved model, trained on all windows (in-sample): a paired description, not an accuracy"}
    out = {
        "build": build, "n_windows": int(len(ds.y)), "folds": len(folds),
        "oof": {"zero_phase_trained_tested_zero_phase": _rec(a, pos), "zero_phase_trained_tested_causal": _rec(b, pos),
                "causal_trained_tested_causal": _rec(c, pos)},
        "ba_drop_causal_vs_zero_phase_fold_mean": float(fold_diff.mean()), "ba_drop_fold_std": float(fold_diff.std(ddof=1)),
        "ba_drop_folds_worse": int(np.sum(fold_diff < 0)), "ba_drop_per_fold": [float(v) for v in fold_diff],
        "paired_saved_model": pair,
        "note": ("Out-of-fold, same grouped folds as M3. `tested on causal features` uses the audio filtered the way the live "
                 "engine filters it. Windows are gated on the zero-phase RMS in both cases."),
    }
    scatter = {"pz": pz[:, 0], "pc": pc[:, 0], "thr": thr}
    return out, recs, scatter


def load_cache_rms(cfg):
    from sentinel.dataset import load_cache
    z = load_cache(cfg, "none")
    i = [str(n) for n in z["feature_names"]].index("rms_mean")
    causal = sc.CausalFeatures(cfg).get("none", "causal", None)
    keep = ~np.isin(z["dataset"].astype(str), cfg["splits"]["holdout_datasets"])
    return z["X"][keep, i].astype(np.float64), causal[keep, i]


def _rec(r, pos: str) -> dict:
    keys = ["ba_fold_mean", "ba_fold_std", "ba_pooled", "ba_lo", "ba_hi", f"recall_{pos}", f"recall_{pos}_lo", f"recall_{pos}_hi",
            f"precision_{pos}", f"auc_{pos}", "specificity_plain", "specificity_hard_negative"]
    return {k: float(r[k]) for k in keys}


# ---- C ---------------------------------------------------------------------------------------------------------------------

def speed(cfg, model, manifest) -> dict:
    sr, need = cfg["audio"]["sr"], cfg["m5"]["check"]["rtf_seconds"]
    air, amb = usable_clips(cfg, manifest, "aircraft"), usable_clips(cfg, manifest, "ambient")
    parts, total, i = [], 0.0, 0
    while total < need:
        for lst in (amb, air):
            x = load_audio(lst[i % len(lst)], sr)
            parts.append(x)
            total += len(x) / sr
        i += 1
    out = sc.engine_speed(model, cfg, np.concatenate(parts))
    out["machine"] = {"platform": platform.platform(), "python": platform.python_version(), "cpu_count": __import__("os").cpu_count()}
    out["note"] = ("Wall time on the machine that ran this, one process. First window includes library warm-up. Real-time factor < 1 "
                   "means the engine keeps up; the window time must stay below the 500 ms hop.")
    return out


# ---- D ---------------------------------------------------------------------------------------------------------------------

def snr_study(cfg, model, manifest) -> tuple[dict, list]:
    ck = cfg["m5"]["check"]
    clips = [load_audio(p, cfg["audio"]["sr"]) for p in spread(usable_clips(cfg, manifest, "aircraft"), int(ck["snr_trials"]))]
    out, pts = {}, []
    for snr in ck["snr_db"]:
        rows = [sc.snr_estimate_errors(model, cfg, c, float(snr), seed=cfg["seed"] + j) for j, c in enumerate(clips)]
        s = sc.summarize_snr_errors(rows)
        s["n_clips"] = len(clips)
        s["share_of_clip_windows_flagged"] = float(np.mean(np.concatenate([r["flagged"] for r in rows])))
        out[f"{snr:+g}"] = s
        pts.append((float(snr), np.concatenate([r["truth"] for r in rows]), np.concatenate([r["estimate"] for r in rows])))
        print(f"[D] nominal {snr:+g} dB: bias {s.get('bias_db', float('nan')):+.1f} dB, MAE {s.get('mae_db', float('nan')):.1f} dB "
              f"over {s['n_windows']} windows", flush=True)
    return {"by_nominal_snr_db": out, "noise": "pink, in-band power 1e-4 (rms 0.01 after the band-pass)", "clips": len(clips),
            "note": ("Truth of a window = power of the clip alone in that window over the noise power, both in-band. The engine's "
                     "noise floor comes only from windows it does not flag, so a clip it misses leaks into the floor.")}, pts


# ---- figures ---------------------------------------------------------------------------------------------------------------

def fig_snr(pts, path: Path) -> Path:
    import matplotlib.pyplot as plt
    plotstyle.apply()
    fig, ax = plt.subplots(figsize=(4.8, 4.4), constrained_layout=True)
    lo, hi = -10, 32
    ax.plot([lo, hi], [lo, hi], color=plotstyle.MUTED, linestyle=":", linewidth=1.0, label="estimate = truth")
    for i, (snr, t, e) in enumerate(pts):
        st = plotstyle.style(i)
        ok = np.isfinite(e)
        ax.plot(t[ok], e[ok], linestyle="none", marker=st["marker"], color=st["color"], markersize=3.5, alpha=0.6,
                markeredgewidth=0, label=f"nominal {snr:+g} dB")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("true in-band SNR of the window (dB)")
    ax.set_ylabel("engine SNR estimate (dB)")
    ax.set_title("Per-window SNR estimate on known mixtures", fontsize=9)
    ax.legend(loc="upper left")
    return plotstyle.save(fig, path)


def fig_causal(recs: pd.DataFrame, scatter: dict, path: Path) -> Path:
    import matplotlib.pyplot as plt
    plotstyle.apply()
    fig, (a, b) = plt.subplots(1, 2, figsize=(9.0, 4.2), constrained_layout=True, gridspec_kw={"width_ratios": [1, 1.1]})
    rng = np.random.default_rng(0)
    n = len(scatter["pz"])
    i = rng.choice(n, min(n, 4000), replace=False)
    f = lambda v: np.clip(v, 1e-4, 1.0)
    a.plot([1e-4, 1], [1e-4, 1], color=plotstyle.MUTED, linestyle=":", linewidth=1.0)
    a.plot(f(scatter["pz"][i]), f(scatter["pc"][i]), linestyle="none", marker="o", markersize=2.5, alpha=0.35,
           color=plotstyle.SERIES[0], markeredgewidth=0)
    a.axvline(scatter["thr"], color=plotstyle.SERIES[1], linestyle="--", linewidth=1.0, label=f"saved threshold {scatter['thr']:.2f}")
    a.axhline(scatter["thr"], color=plotstyle.SERIES[1], linestyle="--", linewidth=1.0)
    a.set_xscale("log")
    a.set_yscale("log")
    a.set_xlabel("probability, zero-phase features")
    a.set_ylabel("probability, causal (live) features")
    a.set_title("Saved model on the same windows (in-sample)", fontsize=9)
    a.legend(loc="upper left")
    rows = [("trained_zero_phase", "zero-phase features", "train zero-phase\ntest zero-phase"),
            ("trained_zero_phase", "causal features", "train zero-phase\ntest causal"),
            ("trained_causal", "causal features", "train causal\ntest causal")]
    for k, (tag, tested, name) in enumerate(rows):
        r = recs[(recs["tag"] == tag) & (recs["tested_on"] == tested)].iloc[0]
        st = plotstyle.style(k)
        b.errorbar([k], [r["ba_pooled"]], yerr=[[r["ba_pooled"] - r["ba_lo"]], [r["ba_hi"] - r["ba_pooled"]]], fmt=st["marker"],
                   color=st["color"], capsize=3, markersize=6, label=name.replace("\n", ", "))
    b.axhline(0.5, color=plotstyle.MUTED, linestyle=":", linewidth=0.8)
    b.set_xticks(range(3), [r[2] for r in rows])
    b.set_xlim(-0.5, 2.5)
    b.set_ylim(0.4, 1.0)
    b.set_ylabel("balanced accuracy (pooled out-of-fold, 95% CI)")
    b.set_title("What the causal filter costs", fontsize=9)
    return plotstyle.save(fig, path)


# ---- main ------------------------------------------------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--force", action="store_true", help="rebuild the causal feature cache")
    ap.add_argument("--quick", action="store_true", help="skip B (the causal feature cache and cross-validation)")
    args = ap.parse_args()
    setup_logging()
    cfg = load_config(args.config)
    set_seeds(cfg["seed"])
    t0 = time.perf_counter()
    try:
        model = load_model(project_path(cfg, "models") / "pipeline.joblib", cfg)
    except FileNotFoundError:
        log.error("models/pipeline.joblib not found. Run: make final")
        return 1
    manifest = pd.read_csv(project_path(cfg, "manifest"))
    tables, figs = project_path(cfg, "tables"), project_path(cfg, "figures")
    res: dict = {}

    air, amb = usable_clips(cfg, manifest, "aircraft"), usable_clips(cfg, manifest, "ambient")
    n = int(cfg["m5"]["check"]["n_clips_invariance"])
    res["chunk_invariance"] = chunk_invariance(cfg, model, spread(air, n // 2) + spread(amb, n - n // 2))
    ci = res["chunk_invariance"]
    print(f"[A] chunk sizes {ci['chunk_sizes_s']} s on {ci['n_clips']} clips: max |dp| = {ci['max_abs_dp']:.2e}, "
          f"events identical: {ci['events_identical']}", flush=True)

    res["speed"] = speed(cfg, model, manifest)
    s = res["speed"]
    print(f"[C] {s['audio_seconds']} s of audio in {s['wall_seconds']} s (real-time factor {s['real_time_factor']}); window "
          f"median {s['window_ms_median']} ms, p95 {s['window_ms_p95']} ms, max {s['window_ms_max']} ms; hop {s['hop_ms']} ms", flush=True)

    res["snr_estimate"], pts = snr_study(cfg, model, manifest)
    rows = [{"nominal_snr_db": float(k), **v} for k, v in res["snr_estimate"]["by_nominal_snr_db"].items()]
    pd.DataFrame(rows).to_csv(tables / "m5_snr_estimate.csv", index=False)
    fig_snr(pts, figs / "m5_snr_estimate.png")

    if not args.quick:
        out, recs, scatter = causal_vs_offline(cfg, model, args)
        res["causal_vs_zero_phase"] = out
        recs.to_csv(tables / "m5_causal_vs_zero_phase.csv", index=False)
        fig_causal(recs, scatter, figs / "m5_causal_vs_zero_phase.png")
        o = out["oof"]
        print("[B] balanced accuracy (pooled out-of-fold, 95% CI):")
        for k, name in (("zero_phase_trained_tested_zero_phase", "train zero-phase, test zero-phase"),
                        ("zero_phase_trained_tested_causal", "train zero-phase, test causal   "),
                        ("causal_trained_tested_causal", "train causal,     test causal   ")):
            print(f"      {name}: {o[k]['ba_pooled']:.3f} [{o[k]['ba_lo']:.3f}, {o[k]['ba_hi']:.3f}]   aircraft recall {o[k]['recall_aircraft']:.3f}")
        print(f"      fold-paired change from zero-phase to causal features: {out['ba_drop_causal_vs_zero_phase_fold_mean']:+.4f} "
              f"(fold std {out['ba_drop_fold_std']:.4f}, worse in {out['ba_drop_folds_worse']} of {out['folds']} folds)")
        p = out["paired_saved_model"]
        print(f"      saved model, same windows: mean |dp| {p['mean_abs_dp']:.3f}, p95 {p['p95_abs_dp']:.3f}, max {p['max_abs_dp']:.3f}; "
              f"alert decision flips {p['alert_decision_flips']} of {p['n_windows']} ({p['alert_flip_fraction']:.1%}); gate flips {p['gate_flips_all_rows']}")
    res["wall_seconds"] = round(time.perf_counter() - t0, 1)
    update_results(cfg, "stream_check", res, merge=args.quick)
    print(f"\nwrote results.json `stream_check`, results/tables/m5_*.csv, results/figures/m5_*.png ({res['wall_seconds']} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
