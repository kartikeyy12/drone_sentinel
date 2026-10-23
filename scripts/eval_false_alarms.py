#!/usr/bin/env python
"""M6 `make eval-fa`: false alarms per hour versus alert threshold, event recall on the scenarios, and the operating point.

Streams scored (the engine runs on real audio; window probabilities are stored and the alert logic is swept over them):
  esc50_crossfit  per M3 fold, the fold's held-out ESC-50 ambient clips joined into one stream, scored by the model of that
                  fold (never saw those recordings). Hard negatives (engine, train, vacuum cleaner, ...) are included and
                  reported separately. These are 5 s clips joined with abrupt cuts: a stress test, not a continuous recording.
  own             long recordings in data/own/ (if any), scored by the saved model. Never seen by training.
  scenarios       data/scenarios/*.wav from `make scenarios`, scored by the model of their fold; give event recall and latency.
The operating point is the alert-ON threshold with at most m6.false_alarms.max_per_hour false alarms per hour (on `own` if it has
at least m6.false_alarms.min_own_hours of audio, otherwise on esc50_crossfit) and the best mean event recall over the scenario SNRs.
Targets are ESC-50 aircraft clips; there is no drone class. Writes results.json `false_alarms`.
"""
from __future__ import annotations

import argparse
import bisect
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sentinel import alerting as al  # noqa: E402
from sentinel import plotstyle  # noqa: E402
from sentinel import scenarios as sc  # noqa: E402
from sentinel.bundle import load_model  # noqa: E402
from sentinel.config import PROJECT_ROOT, load_config, project_path, set_seeds, setup_logging, update_results  # noqa: E402
from sentinel.cv import make_folds  # noqa: E402
from sentinel.dataset import build_dataset  # noqa: E402
from sentinel.engine import run_offline  # noqa: E402
from sentinel.preprocess import load_audio  # noqa: E402

log = logging.getLogger("eval_false_alarms")
SRC_ESC, SRC_OWN = "esc50_crossfit", "own"


def score(model, cfg, x: np.ndarray, target: str, chunk_s: float) -> list[al.Obs]:
    wins, _, _ = run_offline(model, cfg, x, chunk_s=chunk_s, t0_wall=0.0)
    return sc.obs_from_windows(wins, target)


def esc_streams(cfg, models, man, target, chunk_s) -> list[sc.BgStream]:
    sr = cfg["audio"]["sr"]
    out = []
    for f, m in enumerate(models):
        rows = man[(man["label"] == "ambient") & (man["fold"] == f)].sample(frac=1.0, random_state=cfg["seed"] + 2000 + f)
        clips = [(load_audio(PROJECT_ROOT / r.path, sr), r.category, bool(r.hard_negative)) for r in rows.itertuples()]
        x, segs = sc.concat_clips(clips, sr)
        out.append(sc.BgStream(SRC_ESC, score(m, cfg, x, target, chunk_s), len(x) / sr / 3600.0, segs))
        print(f"[esc50_crossfit] fold {f}: {len(clips)} clips, {len(x) / sr / 60:.1f} min", flush=True)
    return out


def own_streams(cfg, model, target, chunk_s) -> list[sc.BgStream]:
    sr, d = cfg["audio"]["sr"], PROJECT_ROOT / cfg["m6"]["false_alarms"]["own_dir"]
    files = sorted(p for p in d.glob("*") if p.suffix.lower() in (".wav", ".flac")) if d.exists() else []
    out = []
    for p in files:
        x = load_audio(p, sr)
        out.append(sc.BgStream(SRC_OWN, score(model, cfg, x, target, chunk_s), len(x) / sr / 3600.0, None))
        print(f"[own] {p.name}: {len(x) / sr / 60:.1f} min", flush=True)
    return out


def scenario_streams(cfg, models, target, chunk_s) -> list[sc.ScenarioStream]:
    d = PROJECT_ROOT / cfg["m6"]["scenarios"]["dir"]
    idx = json.loads((d / "scenarios.json").read_text())
    out = []
    for s in idx["streams"]:
        x = load_audio(d / s["file"], cfg["audio"]["sr"])
        out.append(sc.ScenarioStream(s["snr_db"], score(models[s["fold"]], cfg, x, target, chunk_s), s["truth"]))
    print(f"[scenarios] scored {len(out)} streams", flush=True)
    return out


def fa_breakdown(bgs: list[sc.BgStream], params: al.AlertParams, target: str) -> tuple[pd.DataFrame, dict]:
    """False alarms at the operating point on the ESC-50 streams, by the clip category under the event midpoint."""
    rows, hours = [], {"hard_negative": 0.0, "other_ambient": 0.0}
    for k, b in enumerate(b for b in bgs if b.segments is not None):
        starts = [s.start_s for s in b.segments]
        for s in b.segments:
            hours["hard_negative" if s.hard_negative else "other_ambient"] += (s.end_s - s.start_s) / 3600.0
        for e in al.detect_events(params, b.obs, target):
            s = b.segments[max(0, bisect.bisect_right(starts, (e.t_start + e.t_end) / 2) - 1)]
            rows.append({"stream": k, "t_start_s": e.t_start, "duration_s": e.duration_s, "peak_prob": e.peak_prob,
                         "category": s.category, "hard_negative": s.hard_negative})
    df = pd.DataFrame(rows, columns=["stream", "t_start_s", "duration_s", "peak_prob", "category", "hard_negative"])
    out = {}
    for name, hn in (("hard_negative", True), ("other_ambient", False)):
        k = int((df["hard_negative"] == hn).sum()) if len(df) else 0
        lo, hi = sc.poisson_ci(k, hours[name])
        out[name] = {"false_alarms": k, "hours": hours[name], "per_hour": k / hours[name] if hours[name] else float("nan"),
                     "per_hour_ci95": [lo, hi]}
    return df, out


def figures(table: pd.DataFrame, per: pd.DataFrame, op: dict | None, source: str, figs: Path, srcs: list[str], default_t: float) -> None:
    import matplotlib.pyplot as plt
    plotstyle.apply()
    fig, (a, b) = plt.subplots(1, 2, figsize=(9.6, 4.2), constrained_layout=True)
    for i, s in enumerate(srcs):
        st = plotstyle.style(i)
        a.plot(table["t_on"], table[f"fa_per_hour_{s}"], label=f"{s} ({table[f'hours_{s}'].iloc[0]:.2f} h)", **st)
        a.fill_between(table["t_on"], table[f"fa_per_hour_{s}_lo"], table[f"fa_per_hour_{s}_hi"], color=st["color"], alpha=0.12, linewidth=0)
    a.axhline(1.0, color=plotstyle.MUTED, linestyle=":", linewidth=1.0)
    a.text(table["t_on"].max(), 1.1, "1 per hour", ha="right", fontsize=7, color=plotstyle.MUTED)
    a.axvline(default_t, color=plotstyle.SERIES[3], linestyle="--", linewidth=1.0, label=f"model threshold {default_t:.2f}")
    if op:
        a.axvline(op["t_on"], color=plotstyle.INK, linestyle="-", linewidth=1.0, label=f"operating point {op['t_on']:.2f}")
    a.set_yscale("symlog", linthresh=1.0)
    a.set_ylim(bottom=0)
    a.set_xlabel("alert ON threshold on the smoothed probability")
    a.set_ylabel("false alarms per hour (band: exact 95% CI)")
    a.set_title("False alarms versus threshold", fontsize=9)
    a.legend(loc="upper right")
    pts = sorted({default_t, *(op and [op["t_on"]] or []), 0.1, 0.5})
    for i, t in enumerate(pts):
        d = per[np.isclose(per["t_on"], t)] if np.isclose(per["t_on"], t).any() else per[np.isclose(per["t_on"], per["t_on"].iloc[(per["t_on"] - t).abs().argmin()])]
        d = d.sort_values("snr_db")
        st = plotstyle.style(i)
        b.plot(d["snr_db"], d["event_recall"], label=f"t_on {d['t_on'].iloc[0]:.2f}", **st)
    b.set_xlabel("scenario SNR: clip over background stream (dB)")
    b.set_ylabel("event recall (aircraft clips)")
    b.set_ylim(-0.02, 1.02)
    b.set_title("Event recall versus SNR (cross-fitted)", fontsize=9)
    b.legend(loc="upper left")
    plotstyle.save(fig, figs / "m6_false_alarms_and_recall.png")
    fig, ax = plt.subplots(figsize=(4.8, 4.2), constrained_layout=True)
    fa = table[f"fa_per_hour_{source}"]
    ax.plot(fa, table["event_recall_mean"], label="threshold sweep", **{**plotstyle.style(0), "marker": None})
    if op:
        ax.plot([op[f"fa_per_hour_{source}"]], [op["event_recall_mean"]], linestyle="none", marker="o", color=plotstyle.SERIES[1],
                markersize=7, label=f"operating point t_on {op['t_on']:.2f}")
    ax.set_xscale("symlog", linthresh=1.0)
    ax.set_xlim(left=0)
    ax.set_xlabel(f"false alarms per hour ({source})")
    ax.set_ylabel("mean event recall over scenario SNRs")
    ax.set_title("Trade-off curve", fontsize=9)
    ax.legend(loc="upper left")
    plotstyle.save(fig, figs / "m6_tradeoff.png")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    setup_logging()
    cfg = load_config(args.config)
    set_seeds(cfg["seed"])
    t0 = time.perf_counter()
    mp = project_path(cfg, "models") / "pipeline.joblib"
    if not mp.exists():
        log.error("%s not found. Run: make final", mp)
        return 1
    if not (PROJECT_ROOT / cfg["m6"]["scenarios"]["dir"] / "scenarios.json").exists():
        log.error("no scenarios found. Run: make scenarios")
        return 1
    model = load_model(mp, cfg)
    target = model.targets[0]
    fa, chunk = cfg["m6"]["false_alarms"], float(cfg["m6"]["false_alarms"]["chunk_s"])
    ds = build_dataset(cfg, "none")
    folds = make_folds(ds.y, ds.groups, cfg["splits"]["cv_folds"], cfg["seed"])
    man = pd.read_csv(project_path(cfg, "manifest"))
    man["fold"] = sc.manifest_folds(man, ds, folds)
    man = man[~man["dataset"].isin(cfg["splits"]["holdout_datasets"])]
    models = sc.fold_models(cfg, model, ds, folds)
    bgs = esc_streams(cfg, models, man, target, chunk) + own_streams(cfg, model, target, chunk)
    scen = scenario_streams(cfg, models, target, chunk)
    hours = {s: sum(b.hours for b in bgs if b.source == s) for s in sorted({b.source for b in bgs})}
    source = SRC_OWN if hours.get(SRC_OWN, 0.0) >= fa["min_own_hours"] else SRC_ESC

    sw = fa["sweep"]
    default_t = float(model.thresholds[target])
    grid = sc.threshold_grid(sw["t_on_min"], sw["t_on_max"], sw["step"], [default_t])
    table, per = sc.sweep(bgs, scen, grid, cfg["m5"]["alert"], target, cfg["m6"]["scenarios"]["match_tol_s"],
                          fa["selection_snrs"], default_t)
    op = sc.choose_operating_point(table, source, fa["max_per_hour"])

    tables, figs = project_path(cfg, "tables"), project_path(cfg, "figures")
    table.to_csv(tables / "m6_fa_vs_threshold.csv", index=False)
    per.to_csv(tables / "m6_scenario_recall.csv", index=False)
    figures(table, per, op, source, figs, sorted(hours), default_t)

    res: dict = {"target_class": target, "hours": hours, "selection_source": source, "max_false_alarms_per_hour": fa["max_per_hour"],
                 "default_t_on": default_t, "n_scenario_streams": len(scen),
                 "clips_per_snr": int(per[per["t_on"] == per["t_on"].iloc[0]].groupby("snr_db")["n_clips"].first().iloc[0])}
    for name, t in (("at_model_threshold", default_t), ("at_operating_point", op["t_on"] if op else None)):
        if t is None:
            res[name] = None
            continue
        r = table.iloc[int((table["t_on"] - t).abs().argmin())].to_dict()        # t is a value of the grid
        res[name] = {"t_on": float(t), "t_off": float(r["t_off"]), "event_recall_mean": float(r["event_recall_mean"]),
                     "false_alarms": {s: {"count": int(r[f"fa_count_{s}"]), "hours": float(r[f"hours_{s}"]),
                                          "per_hour": float(r[f"fa_per_hour_{s}"]),
                                          "per_hour_ci95": [float(r[f"fa_per_hour_{s}_lo"]), float(r[f"fa_per_hour_{s}_hi"])]} for s in hours},
                     "by_snr": {f"{x.snr_db:+g}": {"event_recall": float(x.event_recall), "detected": int(x.detected), "clips": int(x.n_clips),
                                                   "latency_median_s": float(x.latency_median_s), "latency_p90_s": float(x.latency_p90_s),
                                                   "false_events_in_scenarios": int(x.false_events_in_scenarios)}
                                for x in per[np.isclose(per["t_on"], t)].itertuples()}}
    if op:
        df, br = fa_breakdown(bgs, sc.params_for(cfg["m5"]["alert"], op["t_on"], default_t), target)
        df.to_csv(tables / "m6_false_alarm_events.csv", index=False)
        res["at_operating_point"]["esc50_false_alarms_by_clip_type"] = br
    budgets = {}
    for b in (1.0, 5.0, 10.0, 30.0, 100.0):
        o = sc.choose_operating_point(table, source, b)
        budgets[f"{b:g}"] = None if o is None else {"t_on": o["t_on"], "false_alarms_per_hour": o[f"fa_per_hour_{source}"],
                                                    "event_recall_mean": o["event_recall_mean"]}
    res["recall_by_false_alarm_budget"] = budgets
    res["note"] = ("Targets are ESC-50 aircraft clips; there is no drone class. esc50_crossfit false alarms come from joined 5 s ESC-50 "
                   "clips scored by models that never saw them; they are not long continuous recordings. Rates have wide exact Poisson "
                   "intervals when the hours are few. t_on acts on the smoothed probability inside the alert logic (m5.alert), "
                   "not on the window-level threshold of M4. At low scenario SNR part of the event recall is chance: the background itself raises "
                   "false alarms at low thresholds, and an alarm within match_tol_s of a clip counts for it. To use the point: set m5.alert.t_on in configs/default.yaml.")
    res["wall_seconds"] = round(time.perf_counter() - t0, 1)
    update_results(cfg, "false_alarms", res)

    print(f"\nAudio scored for false alarms: " + ", ".join(f"{s} {h:.2f} h" for s, h in hours.items()) + f"; operating point chosen on {source}")
    if source != SRC_OWN:
        print("NOTE: no (or too little) audio in data/own, so the false-alarm rate comes from ESC-50 clips only and the "
              f"{fa['max_per_hour']:g}-per-hour target cannot be validated on real long recordings.")
    for name in ("at_model_threshold", "at_operating_point"):
        r = res.get(name)
        if r is None:
            print(f"{name}: no threshold keeps false alarms at or below {fa['max_per_hour']:g} per hour on {source}")
            continue
        f = r["false_alarms"][source]
        print(f"{name}: t_on {r['t_on']:.3f}: {f['count']} false alarms in {f['hours']:.2f} h = {f['per_hour']:.2f}/h "
              f"(95% CI {f['per_hour_ci95'][0]:.2f} to {f['per_hour_ci95'][1]:.2f}); mean event recall {r['event_recall_mean']:.3f}")
        print("   event recall by SNR: " + ", ".join(f"{k} dB {v['event_recall']:.2f} ({v['detected']}/{v['clips']}, latency {v['latency_median_s']:.1f} s)"
                                                    for k, v in r["by_snr"].items()))
    print("false-alarm budget -> threshold, mean event recall: " + "; ".join(
        f"{k}/h: " + ("none" if v is None else f"t_on {v['t_on']:.2f}, recall {v['event_recall_mean']:.3f}") for k, v in budgets.items()))
    print(f"\nwrote results.json `false_alarms`, results/tables/m6_*.csv, results/figures/m6_*.png ({res['wall_seconds']} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
