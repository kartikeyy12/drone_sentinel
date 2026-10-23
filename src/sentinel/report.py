"""Single-file HTML report: runs/<timestamp>/report.html with every figure embedded as base64 PNG.

Every number comes from results/results.json or from the run directory (run.json, events.csv, windows.csv); nothing is typed in.
The two run-specific figures (probability timeline, spectrogram) are drawn here; the rest are the PNGs written by M1 to M7.
"""
from __future__ import annotations

import base64
import csv
import html
import io
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from scipy import signal

from . import plotstyle as ps
from .config import config_hash, project_path

FIG_MISSING = "figure not found (run the make target that creates it)"
SNR_CAVEAT = ("SNR est. is a power-subtraction estimate against the 10th percentile of recent frame power. It is meaningful only when the background has a "
              "steady noise floor. On ESC-50 streams (clips of very different levels joined together, some padded with digital silence) the floor is pulled far below "
              "the typical background and the estimate is much too high, so do not quote it from such a run.")


# ---- run directory -----------------------------------------------------------------------------------------------------------

def latest_run(runs_dir: Path) -> Path | None:
    """Newest run directory that has a windows.csv and a run.json (names are UTC timestamps, so the sort is chronological)."""
    cands = [d for d in sorted(Path(runs_dir).glob("*")) if (d / "run.json").exists() and (d / "windows.csv").exists()]
    return cands[-1] if cands else None


def _epoch(iso: str) -> float:
    return datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp()


def read_run(run_dir: Path) -> dict[str, Any]:
    """meta, windows (dict of arrays) and events (list of dicts with t0_s/t1_s relative to the stream start)."""
    meta = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    with open(run_dir / "windows.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    pcols = [c for c in (rows[0].keys() if rows else []) if c.startswith("p_")]
    wins = {"t_start": np.array([float(r["t_start"]) for r in rows]), "t_end": np.array([float(r["t_end"]) for r in rows]),
            "gated": np.array([int(r["gated"]) for r in rows], bool),
            "snr_db": np.array([float(r["snr_db"]) if r["snr_db"] != "" else np.nan for r in rows])}
    for c in pcols:
        wins[c] = np.array([float(r[c]) for r in rows])
        wins["smoothed_" + c[2:]] = np.array([float(r["smoothed_" + c[2:]]) for r in rows])
    t0 = _epoch(meta["started_at"])
    events = []
    with open(run_dir / "events.csv", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            r["t0_s"], r["t1_s"] = _epoch(r["t_start"]) - t0, _epoch(r["t_end"]) - t0
            events.append(r)
    return {"meta": meta, "windows": wins, "events": events, "targets": [c[2:] for c in pcols]}


# ---- figures drawn per run -----------------------------------------------------------------------------------------------------

def _png_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=ps.DPI)
    ps.plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def timeline_png(run: Mapping[str, Any]) -> str:
    """Window probability and its EMA per target class with the alert threshold, closed events shaded."""
    ps.apply()
    w, meta = run["windows"], run["meta"]
    fig, ax = ps.plt.subplots(figsize=(9, 3.2))
    for i, t in enumerate(run["targets"]):
        mid = (w["t_start"] + w["t_end"]) / 2
        ax.plot(mid, w["p_" + t], label=f"{t} p (window)", linewidth=1.0, **{**ps.style(i), "marker": None})
        ax.plot(mid, w["smoothed_" + t], label=f"{t} smoothed", linewidth=1.8, color=ps.SERIES[(i + 1) % len(ps.SERIES)], linestyle="-")
        thr = meta.get("alert_params", {}).get(t, {}).get("t_on")
        if thr is not None:
            ax.axhline(thr, color=ps.MUTED, linestyle=":", linewidth=1.2)
            ax.text(mid[0], thr + 0.015, f"t_on {thr:.2f}", color=ps.MUTED, fontsize=8)
    for k, e in enumerate(run["events"]):
        ax.axvspan(e["t0_s"], e["t1_s"], color=ps.SERIES[3], alpha=0.25, label="closed event" if k == 0 else None)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("stream time (s)")
    ax.set_ylabel("probability")
    ax.set_title("Class probability timeline, events shaded")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.22))
    fig.tight_layout()
    return _png_b64(fig)


def spectrogram_png(x: np.ndarray, sr: int, events: list[Mapping[str, Any]]) -> str:
    ps.apply()
    f, t, S = signal.spectrogram(x, fs=sr, nperseg=512, noverlap=256, scaling="spectrum")
    db = 10 * np.log10(S + 1e-12)
    fig, ax = ps.plt.subplots(figsize=(9, 3.4))
    im = ax.pcolormesh(t, f, db, shading="auto", cmap="viridis", vmin=db.max() - 80, vmax=db.max())
    for k, e in enumerate(events):
        ax.axvline(e["t0_s"], color="white", linewidth=1.4)
        ax.axvline(e["t1_s"], color="white", linewidth=1.4, linestyle="--")
    ax.grid(False)
    ax.set_xlabel("stream time (s)")
    ax.set_ylabel("frequency (Hz)")
    ax.set_title("Spectrogram (white lines: event start solid, event end dashed)")
    fig.colorbar(im, ax=ax, label="dB")
    fig.tight_layout()
    return _png_b64(fig)


def file_b64(path: Path) -> str | None:
    return base64.b64encode(path.read_bytes()).decode() if path.exists() else None


# ---- formatting ---------------------------------------------------------------------------------------------------------------

def num(x: Any, nd: int = 3, plus: bool = False) -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "n/a"
    return f"{x:+.{nd}f}" if plus else f"{x:.{nd}f}"


def esc(x: Any) -> str:
    return html.escape("" if x is None else str(x))


def table(headers: list[str], rows: list[list[Any]], cls: str = "") -> str:
    th = "".join(f"<th>{esc(h)}</th>" for h in headers)
    tr = "".join("<tr>" + "".join(f"<td>{esc(c)}</td>" for c in r) + "</tr>" for r in rows)
    return f'<table class="{cls}"><thead><tr>{th}</tr></thead><tbody>{tr}</tbody></table>'


def img(b64: str | None, alt: str, caption: str = "") -> str:
    if b64 is None:
        return f'<p class="missing">{esc(alt)}: {FIG_MISSING}</p>'
    cap = f"<figcaption>{esc(caption)}</figcaption>" if caption else ""
    return f'<figure><img alt="{esc(alt)}" src="data:image/png;base64,{b64}">{cap}</figure>'


# ---- sections -----------------------------------------------------------------------------------------------------------------

def sec_summary(run, res, cfg) -> str:
    m = run["meta"]
    fm = res.get("final_model", {}).get("model", {})
    rows = [["source", m.get("source")], ["sensor id", m.get("sensor_id")], ["status", m.get("status")],
            ["started (UTC)", m.get("started_at")], ["stream length (s)", m.get("stream_seconds")], ["windows (gated)", f"{m.get('n_windows')} ({m.get('n_gated_windows')})"],
            ["events", m.get("n_events")], ["dropped short events", m.get("n_dropped_short_events")],
            ["model version (config hash used by the model)", m.get("model_version")], ["pipeline hash", m.get("pipeline_hash")],
            ["config hash now (configs/default.yaml)", config_hash(cfg)],
            ["model classes / targets", f"{fm.get('classes')} / {m.get('targets')}"],
            ["saved window threshold", ", ".join(f"{k} {v:.3f}" for k, v in (m.get("thresholds") or {}).items())],
            ["alert parameters", json.dumps(m.get("alert_params"))], ["silence gate (in-band rms)", m.get("silence_gate_rms")],
            ["median window compute (ms)", m.get("median_window_ms")], ["ZeroMQ", json.dumps(m.get("zmq"))]]
    same = "yes" if m.get("model_version") == config_hash(cfg) else "no (the config changed after the model was built; the model hash is the one that counts)"
    return (f"<h2>1. Run summary</h2>{table(['field', 'value'], rows, 'kv')}"
            f"<p class=\"small\">Model hash equals the current config hash: {esc(same)}.</p>")


def sec_events(run) -> str:
    ev = run["events"]
    if not ev:
        return "<h2>4. Event table</h2><p>No event closed in this run.</p>"
    rows = [[e["event_id"], e["label"], e["t_start"], e["t_end"], e["duration_s"], e["peak_prob"], e["mean_prob"], e["top_alt_class"], e["snr_est_db"] or "n/a"] for e in ev]
    return ("<h2>4. Event table</h2>" + table(["id", "class", "start (UTC)", "end (UTC)", "duration (s)", "peak p", "mean p", "top alt class", "SNR est. (dB)"], rows)
            + f"<p class=\"small\">{SNR_CAVEAT}</p>")


def sec_false_alarms(res, figs) -> str:
    fa = res.get("false_alarms")
    if not fa:
        return "<h2>5. False alarms per hour</h2><p class=\"missing\">results.json has no false_alarms section (run make eval-fa).</p>"
    budget = fa["recall_by_false_alarm_budget"]
    rows = [[f"{b}/h", num(v["t_on"], 2), num(v["false_alarms_per_hour"], 1), num(v["event_recall_mean"], 3)] for b, v in sorted(budget.items(), key=lambda kv: float(kv[0]))]
    op, mt = fa["at_operating_point"], fa["at_model_threshold"]
    sel = fa.get("selection_source", "esc50_crossfit")
    src = ", ".join(f"{k} {v:.2f} h" for k, v in fa["hours"].items())
    ci = op["false_alarms"][sel]["per_hour_ci95"]
    own = "yes" if fa["hours"].get("own", 0) > 0 else "no"
    return ("<h2>5. False alarms per hour versus threshold</h2>"
            f"<p>Audio scored for false alarms: {esc(src)}. Real long recordings of the user (data/own): <b>{own}</b>. "
            f"At t_on {num(mt['t_on'], 3)} (the model threshold): {num(mt['false_alarms'][sel]['per_hour'], 1)} false alarms per hour, mean event recall {num(mt['event_recall_mean'], 3)}. "
            f"At the operating point t_on {num(op['t_on'], 2)} (at most {num(fa['max_false_alarms_per_hour'], 0)} false alarm per hour, best mean event recall): "
            f"{op['false_alarms'][sel]['count']} false alarms in {num(op['false_alarms'][sel]['hours'], 2)} h, 95% upper bound {num(ci[1], 1)} per hour, "
            f"mean event recall {num(op['event_recall_mean'], 3)}.</p>"
            + table(["false-alarm budget", "t_on", "measured false alarms/h", "mean event recall"], rows)
            + img(figs.get("m6_false_alarms_and_recall"), "false alarms and recall versus threshold") + img(figs.get("m6_tradeoff"), "trade-off curve"))


def sec_snr(res, figs) -> str:
    rb = res.get("robustness", {})
    rows = []
    for r in rb.get("records", []):
        if r.get("tag") == "clean_trained" and r.get("variant") in ("none", "wiener", "svd_elbow") and r.get("kind") in ("clean", "white", "pink", "ambient"):
            rows.append([r["variant"], r["kind"], "clean" if r["snr_db"] is None else f"{r['snr_db']:+g}", num(r["ba_fold_mean"], 3), num(r["recall_aircraft"], 3), num(r["specificity_plain"], 3)])
    return ("<h2>6. Accuracy versus SNR</h2>"
            f"<p>Noise is mixed in before any normalisation, at a fixed absolute level per SNR. SNR is nominal against the median aircraft clip (in-band power {num(rb.get('reference', {}).get('p_ref'), 5)}). "
            "Models trained on clean audio, grouped five-fold CV.</p>"
            + img(figs.get("m4_robustness_ba"), "balanced accuracy versus SNR") + img(figs.get("m4_robustness_components"), "recall and specificity versus SNR")
            + "<details><summary>numbers</summary>" + table(["variant", "noise", "SNR (dB)", "balanced accuracy", "aircraft recall", "specificity (plain ambient)"], rows) + "</details>")


def sec_model(res, figs) -> str:
    cv = res["cv_default"]["fold_mean_std"]
    pooled = res["cv_default"]["pooled_oof"]
    ab = res["ablation"]
    rows = []
    for tag, r in [("default pipeline (saved model)", ab["grid_default_pipeline"][0])] + [(f"grid rank {i + 1}", r) for i, r in enumerate(ab["grid_top10"][:5])]:
        rows.append([tag, r["variant"], "none" if r["pca"] is None else r["pca"], r["kernel"], r["C"], f"{num(r['balanced_accuracy_mean'], 3)} ± {num(r['balanced_accuracy_std'], 3)}",
                     num(r["recall_aircraft_mean"], 3), num(r["specificity_hard_negative_mean"], 3)])
    nest = ab["nested"]
    return ("<h2>7. Confusion matrix, explained variance and ablation</h2>"
            f"<p>Saved model: untuned default, variant {esc(res['final_model']['final_config']['variant'])}, PCA {esc(res['final_model']['final_config']['pca'])}, RBF SVM C=1, class weight balanced. "
            f"Grouped CV balanced accuracy {num(cv['balanced_accuracy_mean'], 3)} ± {num(cv['balanced_accuracy_std'], 3)}, aircraft recall {num(cv['recall_aircraft_mean'], 3)}, "
            f"aircraft precision {num(cv['precision_aircraft_mean'], 3)} (the aircraft class is {num(100 * res['cv_default']['dataset']['windows']['aircraft'] / res['cv_default']['dataset']['n_used'], 1)}% of windows), "
            f"specificity on hard negatives {num(cv['specificity_hard_negative_mean'], 3)}. Pooled out-of-fold AUC {num(pooled['auc_aircraft'], 3)}. "
            f"Selecting the best grid cell is optimistic: grid best {num(ab['grid_best']['balanced_accuracy_mean'], 3)}, nested CV {num(nest['pooled_balanced_accuracy'], 3)} pooled "
            f"(optimism gap {num(nest['optimism_gap_best_grid_minus_nested'], 3)}).</p>"
            + img(figs.get("m3_confusion_default"), "confusion matrix of the default pipeline") + img(figs.get("pca_variance"), "PCA explained variance")
            + img(figs.get("pca_scatter"), "PCA projection by class") + img(figs.get("m3_ablation"), "ablation grid")
            + table(["row", "denoising", "PCA", "kernel", "C", "balanced accuracy", "aircraft recall", "hard-negative specificity"], rows)
            + f"<p class=\"small\">{ab['n_configs']} configurations, same grouped folds for all. Full table: results/tables/m3_grid_summary.csv.</p>")


def sec_range(res, figs) -> str:
    r = res.get("range")
    if not r:
        return "<h2>8. Detection range</h2><p class=\"missing\">results.json has no range section (run make range).</p>"
    snr = "<b>not reached</b>" if not r["snr_min_reached"] else f"{num(r['snr_min_db'], 1, True)} dB"
    cs = r["curve_summary"][r["primary_curve"]]
    miss = "".join(f"<li>{esc(m)}</li>" for m in r["range_estimate_missing_inputs"])
    est = ("<p><b>No range estimate is given.</b> Inputs missing:</p><ul>" + miss + "</ul>") if r["range_estimate_m"] is None else f"<p>Range estimate: {num(r['range_estimate_m'], 1)} m.</p>"
    simp = "".join(f"<li>{esc(s)}</li>" for s in r["simplifications"])
    b = res.get("bench", {})
    if b.get("available"):
        brow = [[x["distance_m"], num(x["snr_db"], 1, True), num(x["snr_pred_db"], 1, True), num(x["window_detection_rate"], 2), x["n_active_windows"]] for x in b["rows"]]
        bench = ("<h3>Bench distance test</h3>" + table(["distance (m)", "measured SNR (dB)", "predicted SNR (dB)", "window detection rate", "playback windows"], brow)
                 + img(figs.get("m7_bench"), "bench measurement") + f"<p class=\"small\">{esc(b['note'])}</p>")
    else:
        bench = "<h3>Bench distance test</h3><p>No bench recordings were made. Nothing in this report is a measured distance.</p>"
    return ("<h2>8. Detection range (model estimate, not a measurement)</h2>"
            f"<p>Condition: L_src − 20 log10(d/d0) − α·d − L_ambient ≥ snr_min. snr_min is the lowest SNR from which recall stays at or above {num(r['recall_target'], 2)} on the curve "
            f"<code>{esc(r['primary_curve'])}</code> ({esc(r['primary_curve_axis'])}). Result: snr_min is {snr}; the best recall on that curve is {num(cs['max_recall'], 3)} at {num(cs['snr_at_max_recall_db'], 0, True)} dB.</p>"
            + est + "<p>The drone source level is not assumed (config <code>m7.drone_source_level_db_at_1m</code>, TODO_CITE). The plot below shows the range for <i>hypothetical</i> required SNRs and source levels only.</p>"
            + img(figs.get("m7_recall_curves"), "recall versus SNR") + img(figs.get("m7_range_sensitivity"), "range sensitivity")
            + f"<p class=\"small\">{esc(r['note'])}</p><h3>Simplifications</h3><ul>{simp}</ul>" + bench)


def sec_latency(res) -> str:
    fm, sp = res.get("final_model", {}), res.get("stream_check", {}).get("speed", {})
    lat, fp = fm.get("latency", {}), fm.get("footprint", {})
    st = lat.get("stages", {})
    rows = [[k, num(v["median_ms"], 2), num(v["p95_ms"], 2), num(v["max_ms"], 2)] for k, v in st.items()]
    return ("<h2>9. Latency and model size</h2>"
            + table(["stage", "median (ms)", "p95 (ms)", "max (ms)"], rows)
            + f"<p>Per 0.5 s hop: median total {num(st['total']['median_ms'], 2)} ms, {num(100 * st['total']['share_of_hop_median'], 2)}% of the hop. Whole streaming engine over {num(sp.get('audio_seconds'), 0)} s of audio: "
              f"real-time factor {num(sp.get('real_time_factor'), 4)} (below 1 keeps up; target below 0.2). Measured on {esc(lat.get('machine', {}).get('platform'))}, {esc(lat.get('machine', {}).get('cpu_count'))} CPUs, one thread.</p>"
            + f"<p>Model on disk: {num(fp.get('joblib_file_bytes', 0) / 1024, 0)} KiB ({' and '.join(str(n) for n in fm.get('model', {}).get('n_support_vectors', []))} support vectors (aircraft, ambient), PCA {fm.get('model', {}).get('n_components')} components from {fm.get('model', {}).get('n_features_in')} features).</p>")


def sec_limitations(res, run) -> str:
    cal = res["final_model"]["model"]["calibrators"]
    prior = next(iter(cal.values()))["fitted_at_prior"]
    d = res["cv_default"]["dataset"]
    fa = res.get("false_alarms", {})
    own = fa.get("hours", {}).get("own", 0) > 0
    ba = res["cv_default"]["fold_mean_std"]["balanced_accuracy_mean"]
    items = [
        "<b>There is no drone class.</b> The data are ESC-50 only; the model separates aircraft (airplane, helicopter) from ambient sounds. No number in this report is a drone result. The embedded 'targets' in the scenario streams are aircraft clips.",
        f"<b>Small and clustered data.</b> {d['groups']['aircraft']} aircraft source groups and {d['groups']['ambient']} ambient groups; a few seconds of each recording. Groups are the Freesound source recordings of the ESC-50 clips, so clips cut from one recording never straddle train and test.",
        f"<b>Probabilities are calibrated for clean ESC-50 audio at ESC-50 prevalence</b> (aircraft prior {num(prior, 4)}). In another environment the probability and the false-alarm rate will differ.",
        f"<b>False alarms come from {'real long recordings and ' if own else ''}ESC-50 clips only.</b> " + ("" if own else "No long recording of the user was available, so the 1 false alarm per hour target is not validated on continuous audio. ")
        + "The ESC-50 stream is clips with abrupt cuts joined together, a stress test, not a continuous recording.",
        f"<b>The window classifier is weak at event level.</b> Grouped-CV balanced accuracy is {num(ba, 3)} on windows, and the event recall at the chosen operating point is "
        + num(fa.get("at_operating_point", {}).get("event_recall_mean"), 3) + ". That is a property of the model and data, not of the alert logic; the alert parameters (EMA, N of M, K, merge gap) were not tuned.",
        "<b>In-sample caveat.</b> The saved model is trained on every ESC-50 window; the numbers in this report come from cross-fitted models that never saw the audio they scored (M6) or from grouped CV (M3, M4). Running the saved model over ESC-50 audio gives optimistic results.",
        "<b>The range model is an estimate.</b> Free field, broadband SNR, no A-weighting, no wind, no ground effect, no reverberation; the source level is not assumed.",
        "<b>The microphone path is " + ("tested only through the file source in this run." if str(run["meta"].get("source", "")).startswith("file") else "not verified by this report.") + "</b> Live capture through sounddevice or the Windows TCP streamer works only if you tested it on your machine.",
        "<b>The event SNR estimate is unreliable on ESC-50 audio.</b> " + SNR_CAVEAT + " It does not enter the detection, the probabilities or any recall or false-alarm number.",
        "<b>Defence use.</b> Datasets whose licence forbids defence or surveillance use (Al-Emadi) are not used.",
    ]
    return "<h2>10. Limitations</h2><ul>" + "".join(f"<li>{i}</li>" for i in items) + "</ul>"


CSS = """
:root{--bg:#fcfcfb;--ink:#0b0b0b;--muted:#52514e;--line:#e4e3df;--accent:#2a78d6;--warn:#8a5a00}
@media (prefers-color-scheme: dark){:root{--bg:#161615;--ink:#f1f0ec;--muted:#a9a8a3;--line:#33322f;--accent:#6aa5ee;--warn:#e0b04a}}
body{background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif;margin:0}
main{max-width:980px;margin:0 auto;padding:24px 16px 64px}
h1{font-size:1.6rem;margin:0 0 4px}h2{font-size:1.2rem;margin:36px 0 8px;border-bottom:1px solid var(--line);padding-bottom:4px}h3{font-size:1rem;margin:20px 0 6px}
table{border-collapse:collapse;width:100%;margin:10px 0;font-size:.85rem;display:block;overflow-x:auto}th,td{border-bottom:1px solid var(--line);padding:4px 8px;text-align:left;white-space:nowrap}
th{color:var(--muted);font-weight:600}table.kv td:nth-child(2){white-space:normal;word-break:break-word}
figure{margin:12px 0}img{max-width:100%;height:auto;background:#fcfcfb;border:1px solid var(--line)}figcaption,.small{color:var(--muted);font-size:.8rem}
.missing{color:var(--warn)}.banner{border-left:4px solid var(--warn);padding:8px 12px;margin:12px 0;background:rgba(237,161,0,.08)}code{font-size:.85em}
"""


def build_html(run_dir: Path, cfg: Mapping[str, Any], res: Mapping[str, Any], audio: np.ndarray | None = None, sr: int = 16000) -> str:
    run = read_run(run_dir)
    root = project_path(dict(cfg), "figures")
    names = ["m6_false_alarms_and_recall", "m6_tradeoff", "m4_robustness_ba", "m4_robustness_components", "m3_confusion_default", "pca_variance", "pca_scatter",
             "m3_ablation", "m7_recall_curves", "m7_range_sensitivity", "m7_bench"]
    figs = {n: file_b64(root / f"{n}.png") for n in names}
    spec = img(spectrogram_png(audio, sr, run["events"]), "spectrogram with events") if audio is not None else \
        "<p class=\"missing\">Spectrogram not drawn: the audio of this run was not available (pass --audio).</p>"
    banner = ("<div class=\"banner\"><b>Scope.</b> Everything here is ESC-50, aircraft versus ambient. There is no drone class in the data, "
              "so none of these numbers is a drone result.</div>")
    parts = [f"<h1>SENTINEL-A acoustic sentinel report</h1><p class=\"small\">run {esc(run_dir.name)}, built {esc(datetime.now().astimezone().isoformat(timespec='seconds'))}</p>", banner,
             sec_summary(run, res, cfg), "<h2>2. Class probability timeline</h2>" + img(timeline_png(run), "probability timeline"),
             "<h2>3. Spectrogram with events</h2>" + spec, sec_events(run), sec_false_alarms(res, figs), sec_snr(res, figs), sec_model(res, figs),
             sec_range(res, figs), sec_latency(res), sec_limitations(res, run)]
    return f"<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width, initial-scale=1\"><title>Sentinel report</title><style>{CSS}</style></head><body><main>{''.join(parts)}</main></body></html>"
