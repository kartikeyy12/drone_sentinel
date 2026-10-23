"""HTML report and report notes: sections present, self-contained, numbers come from results.json, no em dashes."""
from __future__ import annotations

import io
import json
import re

import pytest
import soundfile as sf
from rich.console import Console

from helpers_stream import SR, scene
from sentinel import monitor as mon
from sentinel import report as rp
from sentinel.config import load_config, project_path
from sentinel.notes import build_notes
from sentinel.sources import FileSource

CFG = load_config()
RES_PATH = project_path(CFG, "results_json")
needs_results = pytest.mark.skipif(not RES_PATH.exists(), reason="results/results.json not built yet")


@pytest.fixture(scope="module")
def res():
    return json.loads(RES_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def run_dir(toy, tmp_path_factory):
    cfg, model = toy
    d = tmp_path_factory.mktemp("rep")
    x = scene(8, 10, 8)
    sf.write(d / "scene.wav", x, SR)
    s = mon.run(FileSource(d / "scene.wav", SR, chunk_s=0.25), model, cfg, runs_dir=d / "runs", ui=False, t0_wall=1_760_000_000.0,
                console=Console(file=io.StringIO()))
    return d, __import__("pathlib").Path(s["run_dir"]), x


def test_read_run_gives_relative_event_times(run_dir):
    _, rd, _ = run_dir
    run = rp.read_run(rd)
    assert run["targets"] == ["aircraft"] and len(run["events"]) == 1
    e = run["events"][0]
    assert 6.0 < e["t0_s"] < 14.0 and e["t1_s"] > e["t0_s"] + 2.0          # the toy event sits at 8 to 18 s
    assert len(run["windows"]["p_aircraft"]) == len(run["windows"]["t_start"]) == 51


def test_latest_run_is_the_newest_complete_one(tmp_path):
    for n in ("20261001-000000", "20261002-000000", "20261003-000000"):
        (tmp_path / n).mkdir()
    for n in ("20261001-000000", "20261002-000000"):
        (tmp_path / n / "run.json").write_text("{}")
        (tmp_path / n / "windows.csv").write_text("")
    assert rp.latest_run(tmp_path).name == "20261002-000000"          # the newest folder has no files yet
    assert rp.latest_run(tmp_path / "none") is None


def test_run_figures_are_png(run_dir):
    _, rd, x = run_dir
    run = rp.read_run(rd)
    import base64
    for b in (rp.timeline_png(run), rp.spectrogram_png(x, SR, run["events"])):
        assert base64.b64decode(b)[:8] == b"\x89PNG\r\n\x1a\n"


@needs_results
def test_report_has_every_section_and_is_self_contained(run_dir, res):
    d, rd, x = run_dir
    h = rp.build_html(rd, CFG, res, x, SR)
    for n, title in enumerate(["Run summary", "Class probability timeline", "Spectrogram with events", "Event table", "False alarms per hour",
                               "Accuracy versus SNR", "Confusion matrix", "Detection range", "Latency and model size", "Limitations"], 1):
        assert f"<h2>{n}. " in h and title in h
    assert "There is no drone class" in h and h.count("data:image/png;base64,") >= 8
    assert not re.search(r'(src|href)="https?://', h) and "<script" not in h           # one file, nothing fetched
    text = re.sub(r"base64,[A-Za-z0-9+/=]+", "", h)
    assert not re.search(r">(nan|None|inf)<", text)


@needs_results
def test_report_numbers_are_the_ones_in_results_json(run_dir, res):
    d, rd, x = run_dir
    h = rp.build_html(rd, CFG, res, x, SR)
    text = re.sub(r"base64,[A-Za-z0-9+/=]+", "", h)
    op = res["false_alarms"]["at_operating_point"]
    assert f"t_on {op['t_on']:.2f}" in text and f"mean event recall {op['event_recall_mean']:.3f}" in text
    assert f"{res['cv_default']['fold_mean_std']['balanced_accuracy_mean']:.3f}" in text
    rg = res.get("range")                                  # absent until `make range` has run
    if rg is None:
        assert "run make range" in text
    elif rg["snr_min_reached"]:
        assert f"snr_min is {rg['snr_min_db']:+.1f} dB" in text
    else:
        assert "snr_min is <b>not reached</b>" in text
    assert "do not quote it from such a run" in text and text.count("10th percentile of recent frame power") == 2   # event table note and limitation
    assert rd.name in text and "—" not in text                                       # no em dashes in the report


@needs_results
def test_report_without_audio_says_so(run_dir, res):
    _, rd, _ = run_dir
    assert "Spectrogram not drawn" in rp.build_html(rd, CFG, res, None, SR)


@needs_results
def test_make_report_script_writes_report_html_in_the_run_dir(run_dir):
    import importlib.util
    spec = importlib.util.spec_from_file_location("make_report", project_path(CFG, "results_json").parents[1] / "scripts" / "make_report.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _, rd, _ = run_dir
    assert mod.main(["--run", str(rd)]) == 0
    assert (rd / "report.html").stat().st_size > 100_000
    assert mod.main(["--run", str(rd.parent / "nonexistent")]) == 1


@needs_results
def test_notes_cover_the_seven_sections_with_numbers_from_results(res):
    n = build_notes(res, CFG)
    for k, t in enumerate(["Introduction to PCA, SVD and SVM", "Problem statement", "Implementation details", "Python programming",
                           "Results and discussion", "Conclusion", "References"], 1):
        assert f"## {k}. {t}" in n
    op = res["false_alarms"]["at_operating_point"]
    assert f"t_on {op['t_on']:.2f}" in n and "There is no drone class" in n and "No number below is a drone result" in n
    rg = res.get("range")
    assert ("never met" in n) == (rg is not None and not rg["snr_min_reached"])
    assert "—" not in n and "TODO" not in n.replace("TODO_CITE", "")
    assert "not meaningful on the ESC-50 scenario streams" in n and "—" not in n
    assert "Not checked here" in n and "Checked against the source page" in n
