#!/usr/bin/env python
"""M4 `make final`: fit the final pipeline, calibrate it, save models/pipeline.joblib, measure latency and size.

  1. Grouped cross-validation of the final settings (m4.final.config, default m3.default) gives out-of-fold class scores:
     every window is scored by a model that never saw its recording.
  2. A Platt calibrator per target class is fitted on those scores. Its quality is checked cross-fitted (the calibrator that
     scores a fold is fitted on the other folds' scores) against two baselines.
  3. The alarm threshold is the smallest probability whose false-alarm rate on ambient windows is at most
     m4.operating.target_window_far. The table of other targets, and how well each target transfers to held-out folds
     (threshold from the other folds), is stored too. M6 sets the real operating point.
  4. The final model is fitted on ALL non-holdout windows and saved with its description. The file is reloaded and checked.
  5. Per-window latency of each stage and the model footprint are measured on this machine.

Everything is ESC-50, aircraft versus ambient. The probabilities hold at ESC-50's window prevalence (see calibration.py).
"""
from __future__ import annotations

import argparse
import logging
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sentinel import bundle as bd  # noqa: E402
from sentinel import calibration as cal  # noqa: E402
from sentinel.config import PROJECT_ROOT, load_config, project_path, set_seeds, setup_logging, update_results  # noqa: E402
from sentinel.cv import AMBIENT, group_bootstrap_ci, make_folds  # noqa: E402
from sentinel.dataset import build_dataset  # noqa: E402
from sentinel.m3 import cv_run, select_columns  # noqa: E402
from sentinel.m4plots import plot_calibration, plot_operating_curve  # noqa: E402
from sentinel.preprocess import load_audio, preprocess_clip  # noqa: E402
from sentinel.windowing import window_signal  # noqa: E402

log = logging.getLogger("build_final")


def crossfit_probabilities(s: np.ndarray, y: np.ndarray, fold_id: np.ndarray) -> np.ndarray:
    """Calibrated probability of every window from a calibrator fitted on the OTHER folds' out-of-fold scores."""
    p = np.zeros(len(s))
    for f in np.unique(fold_id):
        mine = fold_id == f
        p[mine] = cal.PlattCalibrator().fit(s[~mine], y[~mine]).predict(s[mine])
    return p


def latency_windows(cfg, manifest: pd.DataFrame, n_windows: int, seed: int) -> np.ndarray:
    """Band-passed 1 s windows from a seeded random draw of clips, n_windows of them."""
    rng = np.random.default_rng(seed)
    sr, a = cfg["audio"]["sr"], cfg["audio"]
    wins = []
    for i in rng.permutation(len(manifest)):
        x = load_audio(PROJECT_ROOT / manifest["path"].iloc[i], sr)
        w, _ = window_signal(preprocess_clip(x, sr, cfg), sr, a["window_s"], a["hop_s"])
        wins.append(w)
        if sum(len(v) for v in wins) >= n_windows:
            break
    allw = np.concatenate(wins)
    return allw[rng.permutation(len(allw))[:n_windows]]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--skip-latency", action="store_true", help="skip the timing run (use while other jobs are using the CPU)")
    args = ap.parse_args()
    setup_logging()
    cfg = load_config(args.config)
    set_seeds(cfg["seed"])
    t0 = time.perf_counter()
    fc, m4, seed = bd.final_config(cfg), cfg["m4"], cfg["seed"]
    try:
        ds = build_dataset(cfg, fc["variant"])
    except FileNotFoundError as e:
        log.error("%s", e)
        return 1
    folds = make_folds(ds.y, ds.groups, cfg["splits"]["cv_folds"], seed)
    X = select_columns(ds, cfg, fc["drop_level_features"])
    run = cv_run(X, ds, folds, fc["pca"], fc["svm"], cfg["m3"]["class_weight"], seed)
    classes, S, fold_id = run["classes"], run["scores"], run["fold_id"]
    agg = run["aggregate"]
    print(f"final settings: variant {fc['variant']}, level features {'dropped' if fc['drop_level_features'] else 'kept'}, "
          f"PCA {fc['pca']}, SVM {fc['svm']}")
    print(f"cross-validated balanced accuracy (fold mean) {agg['balanced_accuracy_mean']:.3f} +/- {agg['balanced_accuracy_std']:.3f}  "
          f"({len(ds.y)} windows, {len(set(ds.groups))} source recordings)")

    # ---- final model ---------------------------------------------------------------------------------------------------
    model = bd.build_model(cfg, ds, S, classes)
    model_path = project_path(cfg, "models") / "pipeline.joblib"
    size = bd.save_model(model, model_path)
    reloaded = bd.load_model(model_path, cfg)
    reload_diff = float(np.max(np.abs(model.probabilities(ds.X) - reloaded.probabilities(ds.X))))
    print(f"saved {model_path.relative_to(PROJECT_ROOT)} ({size / 1024:.0f} KiB) and its .json description; "
          f"reload max |dp| = {reload_diff:.1e}")

    tables, figs = project_path(cfg, "tables"), project_path(cfg, "figures")
    tables.mkdir(parents=True, exist_ok=True)
    amb = ds.y == AMBIENT
    out: dict = {"final_config": fc, "cv_balanced_accuracy_fold_mean": agg["balanced_accuracy_mean"],
                 "cv_balanced_accuracy_fold_std": agg["balanced_accuracy_std"], "targets": {}}

    for c in model.targets:
        j = classes.index(c)
        s, y = S[:, j], ds.y == c
        cal_c = model.calibrators[c]
        thr = model.thresholds[c]

        # ---- calibration quality, cross-fitted -----------------------------------------------------------------------
        p_cf = crossfit_probabilities(s, y, fold_id)
        p_naive = 1 / (1 + np.exp(-s))                       # the raw score pushed through a sigmoid, no fitting
        p_prior = np.full(len(s), y.mean())
        nb = m4["operating"]["calibration_bins"]
        quality = {name: {"brier": cal.brier(p, y), "log_loss": cal.log_loss(p, y), "ece": cal.ece(p, y, nb)}
                   for name, p in (("calibrated (cross-fitted)", p_cf), ("raw score through a sigmoid", p_naive),
                                   ("constant prevalence", p_prior))}
        bins = pd.concat([cal.reliability_table(p, y, nb).assign(kind=name)
                          for name, p in (("calibrated (cross-fitted)", p_cf), ("raw score through a sigmoid", p_naive))])
        bins.to_csv(tables / f"m4_calibration_bins_{c}.csv", index=False)
        plot_calibration(bins, figs / f"m4_calibration_{c}.png", f"{c}: reliability, out-of-fold windows")

        # ---- operating points ----------------------------------------------------------------------------------------
        p_oof = cal_c.predict(s)
        use = y | amb
        curve = cal.far_recall_curve(p_oof[use], y[use])
        curve.to_csv(tables / f"m4_operating_curve_{c}.csv", index=False)
        pooled = []
        for far in m4["operating"]["far_grid"]:
            t = cal.threshold_for_far(p_oof[amb], far)
            pooled.append({"target_far": far, "threshold_probability": t, "far": float(np.mean(p_oof[amb] >= t)),
                           "recall": float(np.mean(p_oof[y] >= t))})
        native = {"far": float(np.mean(s[amb] >= 0.0)), "recall": float(np.mean(s[y] >= 0.0)),
                  "probability_at_score_0": float(cal_c.predict(np.array([0.0]))[0]),
                  "note": "the SVM's own decision boundary (score 0)"}
        cf = cal.crossfit_operating_points(p_oof, y, amb, fold_id, m4["operating"]["far_grid"])
        cf.to_csv(tables / f"m4_operating_crossfit_{c}.csv", index=False)
        cfs = cal.summarize_crossfit(cf)
        pd.DataFrame(pooled).to_csv(tables / f"m4_operating_points_{c}.csv", index=False)
        plot_operating_curve(curve, cfs, figs / f"m4_operating_curve_{c}.png", c, f"{c}: recall against window false-alarm rate")

        pred = np.where(p_oof >= thr, c, AMBIENT)
        ci = group_bootstrap_ci(ds.y, pred, ds.groups, classes, cfg["m3"]["bootstrap"]["n"], seed)
        saved = {"threshold_probability": thr, "target_window_far": model.target_far,
                 "far_all_ambient": float(np.mean(p_oof[amb] >= thr)),
                 "far_plain": float(np.mean(p_oof[amb & ~ds.hard_negative] >= thr)),
                 "far_hard_negative": float(np.mean(p_oof[amb & ds.hard_negative] >= thr)),
                 "recall": float(np.mean(p_oof[y] >= thr)), "recall_ci95": [ci[f"recall_{c}_lo"], ci[f"recall_{c}_hi"]],
                 "note": "pooled out-of-fold windows; the threshold was chosen on the same windows, so the cross-fitted "
                         "table is the honest transfer check"}
        out["targets"][c] = {
            "calibrator": {"a": cal_c.a_, "b": cal_c.b_, "fitted_at_prior": cal_c.prior_},
            "calibration_quality_out_of_fold": quality, "pooled_operating_points": pooled, "native_boundary": native,
            "crossfit_operating_points": cfs.to_dict("records"), "saved_threshold": saved,
            "n_windows": {"target": int(y.sum()), "ambient": int(amb.sum())}}
        print(f"\ncalibration of {c} (out-of-fold, calibrator fitted on the other folds): "
              + ", ".join(f"{k}: Brier {v['brier']:.4f} ECE {v['ece']:.4f}" for k, v in quality.items()))
        print(f"native SVM boundary: false-alarm rate {native['far']:.3f}, recall {native['recall']:.3f}")
        print(f"operating points for {c} (pooled out-of-fold; cross-fitted = threshold from the other folds, applied to the held-out fold):")
        t = pd.DataFrame(pooled).merge(cfs, on="target_far")
        print(t.rename(columns={"threshold_probability": "thr_p", "far": "far_pooled", "recall": "rec_pooled", "far_mean": "far_cf",
                                "far_std": "far_cf_sd", "far_max": "far_cf_max", "recall_mean": "rec_cf", "recall_std": "rec_cf_sd",
                                "recall_min": "rec_cf_min"}).round(3).to_string(index=False))
        print(f"saved threshold p >= {thr:.4f} (target window FAR {model.target_far}): FAR {saved['far_all_ambient']:.3f} "
              f"(plain {saved['far_plain']:.3f}, hard negatives {saved['far_hard_negative']:.3f}), recall {saved['recall']:.3f} "
              f"[{saved['recall_ci95'][0]:.3f}, {saved['recall_ci95'][1]:.3f}]")

    # ---- the saved file scores real clips like the cache does -----------------------------------------------------------
    man = pd.read_csv(project_path(cfg, "manifest"))
    z = np.load(PROJECT_ROOT / cfg["features"]["cache_dir"] / f"features_{fc['variant']}.npz", allow_pickle=False)
    diffs = []
    rng = np.random.default_rng(seed)
    for lab in (model.targets[0], AMBIENT, AMBIENT):
        i = int(rng.choice(np.flatnonzero((man["label"] == lab).to_numpy())))
        sc = reloaded.score_audio(load_audio(PROJECT_ROOT / man["path"].iloc[i], cfg["audio"]["sr"]))
        rows = np.flatnonzero(z["file_index"] == i)
        ref = reloaded.probabilities(z["X"][rows].astype(np.float64))
        ref[z["X"][rows, ds.names.index("rms_mean")] < cfg["m3"]["silence_gate_rms"]] = 0.0
        diffs.append(float(np.max(np.abs(sc["p"] - ref))))
    print(f"\nscore_audio on 3 clips vs the feature cache: max |dp| = {max(diffs):.1e}")
    out["consistency"] = {"reload_max_abs_dp": reload_diff, "score_audio_vs_cache_max_abs_dp": max(diffs)}

    # ---- latency and size ---------------------------------------------------------------------------------------------
    out["model"] = reloaded.describe()
    out["footprint"] = bd.model_footprint(reloaded, model_path)
    if not args.skip_latency:
        from threadpoolctl import threadpool_limits
        lat = m4["latency"]
        wins = latency_windows(cfg, man, lat["n_windows"], seed)
        with threadpool_limits(limits=int(lat["threads"])):
            st = bd.benchmark_stages(reloaded, wins, warmup=int(lat["warmup"]))
            summ = bd.stage_summary(st, cfg["audio"]["hop_s"])
            Xb = np.stack([bd.window_features(w, cfg["audio"]["sr"], cfg["features"]) for w in wins]).astype(np.float64)
            tb = time.perf_counter()
            reloaded.probabilities(Xb)
            batch_ms = 1000 * (time.perf_counter() - tb) / len(Xb)
        out["latency"] = {"n_windows": int(len(wins)), "warmup_windows_discarded": int(lat["warmup"]), "threads": int(lat["threads"]),
                          "stages": summ, "decision_batch_ms_per_window": batch_ms,
                          "machine": {"platform": platform.platform(), "processor": platform.processor(),
                                      "cpu_count": os.cpu_count(), "python": platform.python_version()},
                          "note": "wall time on the machine that ran this, one window at a time, single process; the denoiser is "
                                  "not included (variant none needs none)."}
        pd.DataFrame([{"stage": k, **v} for k, v in summ.items()]).to_csv(tables / "m4_latency.csv", index=False)
        print(f"\nlatency per 1 s window, {lat['threads']} thread(s), {len(wins) - lat['warmup']} timed windows (ms): median / p95 / max")
        for k, v in summ.items():
            print(f"  {k:16s} {v['median_ms']:8.2f} {v['p95_ms']:8.2f} {v['max_ms']:8.2f}")
        print(f"  total (features + decision + calibration) is {100 * summ['total']['share_of_hop_median']:.1f}% of the "
              f"{cfg['audio']['hop_s'] * 1000:.0f} ms hop at the median; batch decision {batch_ms:.3f} ms/window")
    fp = out["footprint"]
    print(f"model file {fp['joblib_file_bytes'] / 1024:.0f} KiB (compressed), arrays {fp['array_bytes_total'] / 1024:.0f} KiB, "
          f"support vectors {fp['n_support_vectors']} x {fp['feature_dim_into_svm']} dims, load {fp['load_seconds']} s")

    out["wall_seconds"] = round(time.perf_counter() - t0, 1)
    out["note"] = ("ESC-50 only, aircraft versus ambient. Probabilities are calibrated at ESC-50's window prevalence. The saved "
                   "threshold is an engineering default; M6 sets the real operating point from a false-alarm target.")
    update_results(cfg, "final_model", out)
    print(f"\nwrote results/results.json `final_model`, results/tables/m4_*.csv, results/figures/m4_calibration_*.png, "
          f"m4_operating_curve_*.png ({out['wall_seconds']} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
