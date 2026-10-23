"""M6 building blocks: cross-fitted models, scenario streams, event matching, Poisson intervals, sweep and operating point."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from helpers_stream import SR, noise, tone_stack
from sentinel import alerting as al
from sentinel import scenarios as sc
from sentinel.config import load_config
from sentinel.engine import WindowResult
from sentinel.noise import inband_power

CFG = load_config()
ALERT = CFG["m5"]["alert"]


def ev(t0, t1, alert=None):
    return al.Event("aircraft", t0, t1, t0 + 2.0 if alert is None else alert, 0.9, 0.8, "ambient", None, 3)


# ---- streams -------------------------------------------------------------------------------------------------------------

def test_place_times_no_overlap_gaps_and_margins():
    for seed in range(50):
        t = sc.place_times(np.random.default_rng(seed), 8, 240.0, 5.0, 12.0, 10.0)
        assert len(t) == 8 and t == sorted(t) and t[0] >= 10.0 and t[-1] + 5.0 <= 230.0 + 1e-9
        assert all(b - a >= 5.0 + 12.0 - 1e-9 for a, b in zip(t, t[1:]))
    a = sc.place_times(np.random.default_rng(1), 3, 100.0, 5.0, 12.0, 10.0)
    assert a == sc.place_times(np.random.default_rng(1), 3, 100.0, 5.0, 12.0, 10.0)
    with pytest.raises(ValueError, match="do not fit"):
        sc.place_times(np.random.default_rng(0), 8, 100.0, 5.0, 12.0, 10.0)


def test_concat_clips_segments_and_truncation():
    clips = [(np.ones(2 * SR, np.float32), "rain", False), (np.zeros(3 * SR, np.float32), "engine", True),
             (np.ones(SR, np.float32), "wind", False)]
    x, segs = sc.concat_clips(clips, SR)
    assert len(x) == 6 * SR
    assert [(s.start_s, s.end_s, s.category, s.hard_negative) for s in segs] == [
        (0.0, 2.0, "rain", False), (2.0, 5.0, "engine", True), (5.0, 6.0, "wind", False)]
    x, segs = sc.concat_clips(clips, SR, max_s=4.0)
    assert len(x) == 4 * SR and [s.category for s in segs] == ["rain", "engine"] and segs[-1].end_s == 4.0


@pytest.mark.parametrize("snr", [-5.0, 0.0, 12.0])
def test_embed_sets_the_clip_power_relative_to_the_background(snr):
    bg = noise(20.0, 0.03, seed=1)
    clip = tone_stack(3.0, seed=2)
    mix = sc.embed(bg, [clip], [5.0], snr, SR, CFG)
    d = mix.astype(np.float64) - bg
    a = 5 * SR
    assert np.abs(d[:a]).max() < 1e-6 and np.abs(d[a + len(clip):]).max() < 1e-6          # float32 rounding only
    g = float(np.dot(d[a:a + len(clip)], clip) / np.dot(clip, clip))                      # the gain that was applied
    assert 10 * np.log10(inband_power(g * clip, SR, CFG) / inband_power(bg, SR, CFG)) == pytest.approx(snr, abs=0.05)


def test_embed_rejects_a_clip_that_runs_past_the_end():
    with pytest.raises(ValueError, match="past the end"):
        sc.embed(noise(5.0), [tone_stack(3.0)], [4.0], 0.0, SR, CFG)


# ---- matching ------------------------------------------------------------------------------------------------------------

def test_match_events_detection_latency_and_false_alarms():
    truth = [{"t_start": 20.0, "t_end": 25.0}, {"t_start": 60.0, "t_end": 65.0}, {"t_start": 100.0, "t_end": 105.0}]
    events = [ev(21.0, 26.0, alert=23.5), ev(40.0, 45.0), ev(66.0, 70.0, alert=68.0), ev(25.8, 28.0, alert=27.0)]
    m = sc.match_events(truth, events, tol_s=1.0)
    assert m["detected"] == [True, True, False]
    assert m["latency_s"][0] == pytest.approx(3.5) and m["latency_s"][1] == pytest.approx(8.0) and np.isnan(m["latency_s"][2])
    # 40 to 45 matches nothing; 25.8 to 28 starts within 1 s of the end of clip 1 (25), so it is not a false alarm
    assert [(e.t_start, e.t_end) for e in m["false_events"]] == [(40.0, 45.0)]
    m0 = sc.match_events(truth, events, tol_s=0.0)
    assert m0["detected"] == [True, False, False] and len(m0["false_events"]) == 3


def test_obs_from_windows():
    w = WindowResult(3, 1.5, 2.5, {"aircraft": 0.7}, {"aircraft": 0.6}, False, "aircraft", (), 0.1, 3.0, 1e-4, 5.0)
    o = sc.obs_from_windows([w], "aircraft")[0]
    assert (o.t_start, o.t_end, o.p, o.floor) == (1.5, 2.5, 0.7, 1e-4)


def test_poisson_ci_matches_known_exact_intervals():
    lo, hi = sc.poisson_ci(0, 1.0)
    assert lo == 0.0 and hi == pytest.approx(3.6889, abs=1e-3)
    lo, hi = sc.poisson_ci(1, 1.0)
    assert lo == pytest.approx(0.0253, abs=1e-3) and hi == pytest.approx(5.5716, abs=1e-3)
    lo, hi = sc.poisson_ci(10, 2.0)
    assert lo == pytest.approx(4.795 / 2, abs=2e-3) and hi == pytest.approx(18.390 / 2, abs=2e-3)
    assert np.isnan(sc.poisson_ci(1, 0.0)[0])


# ---- sweep and operating point ----------------------------------------------------------------------------------------------

def _obs(p):
    return [al.Obs(i * 0.5, i * 0.5 + 1.0, float(x)) for i, x in enumerate(p)]


def _synthetic():
    rng = np.random.default_rng(0)
    p = rng.random(7200) * 0.1                              # one hour of low-probability windows
    for k, h in enumerate((0.9, 0.7, 0.5, 0.35)):           # four false alarm bursts of falling height
        p[1000 * (k + 1):1000 * (k + 1) + 12] = h
    bg = [sc.BgStream("esc50_crossfit", _obs(p), 1.0)]
    truth = [{"t_start": 20.0, "t_end": 25.0}, {"t_start": 60.0, "t_end": 65.0}]
    sp = np.zeros(300)
    sp[38:48] = 0.8                                         # clip at 20 s to 25 s
    sp[118:128] = 0.4                                       # a faint clip at 60 s
    return bg, [sc.ScenarioStream(10.0, _obs(sp), truth), sc.ScenarioStream(0.0, _obs(np.zeros(300)), truth)]


def test_sweep_counts_false_alarms_per_hour_and_event_recall():
    bg, scen = _synthetic()
    t, per = sc.sweep(bg, scen, [0.2, 0.45, 0.6, 0.8, 0.95], ALERT, "aircraft", 1.0)
    assert t["fa_count_esc50_crossfit"].tolist() == [4, 3, 2, 1, 0]
    assert t["fa_per_hour_esc50_crossfit"].tolist() == [4.0, 3.0, 2.0, 1.0, 0.0]
    lo, hi = t.loc[t["t_on"] == 0.95, ["fa_per_hour_esc50_crossfit_lo", "fa_per_hour_esc50_crossfit_hi"]].iloc[0]
    assert lo == 0.0 and hi == pytest.approx(3.689, abs=1e-3)
    r10 = per[per["snr_db"] == 10.0].set_index("t_on")
    assert r10.loc[0.2, "event_recall"] == 1.0 and r10.loc[0.45, "event_recall"] == 0.5 and r10.loc[0.95, "event_recall"] == 0.0
    assert (per[per["snr_db"] == 0.0]["event_recall"] == 0.0).all()
    assert r10.loc[0.2, "latency_median_s"] > 0 and np.isnan(r10.loc[0.95, "latency_median_s"])
    row = t.set_index("t_on")
    assert row.loc[0.2, "event_recall_mean"] == pytest.approx(0.5)                # mean of 1.0 (10 dB) and 0.0 (0 dB)
    assert row.loc[0.2, "t_off"] == pytest.approx(0.2 * ALERT["t_off_ratio"])


def test_sweep_selection_snrs_restrict_the_recall_average():
    bg, scen = _synthetic()
    t, _ = sc.sweep(bg, scen, [0.2], ALERT, "aircraft", 1.0, selection_snrs=[10.0])
    assert t["event_recall_mean"].iloc[0] == 1.0


def test_sweep_with_two_sources_keeps_them_separate():
    bg, scen = _synthetic()
    bg.append(sc.BgStream("own", _obs(np.full(720, 0.01)), 0.1))
    t, _ = sc.sweep(bg, scen, [0.2], ALERT, "aircraft", 1.0)
    assert t["fa_count_own"].iloc[0] == 0 and t["hours_own"].iloc[0] == pytest.approx(0.1)
    assert t["fa_count_esc50_crossfit"].iloc[0] == 4


def test_choose_operating_point_rule():
    t = pd.DataFrame({"t_on": [0.2, 0.4, 0.6, 0.8], "fa_per_hour_own": [5.0, 1.0, 0.5, 0.0], "event_recall_mean": [0.9, 0.6, 0.6, 0.2]})
    assert sc.choose_operating_point(t, "own", 1.0)["t_on"] == 0.6                 # ties on recall go to the higher threshold
    op = sc.choose_operating_point(t, "own", 0.0)
    assert op["t_on"] == 0.8 and op["event_recall_mean"] == 0.2
    assert sc.choose_operating_point(t.assign(fa_per_hour_own=[5.0, 4.0, 3.0, 2.0]), "own", 1.0) is None


def test_params_for_derives_t_off_and_validates():
    p = sc.params_for(ALERT, 0.5, 0.3)
    assert p.t_on == 0.5 and p.t_off == pytest.approx(0.5 * ALERT["t_off_ratio"])
    with pytest.raises(ValueError):
        sc.params_for(ALERT, 1.5, 0.3)


# ---- cross-fitting -----------------------------------------------------------------------------------------------------------

def test_fold_models_are_fitted_on_their_training_windows_only(toy_full):
    cfg, model, ds, folds = toy_full
    models = sc.fold_models(cfg, model, ds, folds)
    assert len(models) == len(folds)
    for m, (tr, te) in zip(models, folds):
        pipe = m.pipeline
        np.testing.assert_allclose(pipe.mu_, ds.X[tr][:, pipe.columns_].mean(axis=0))      # z-score statistics come from the train rows
        assert pipe.classes_ == model.classes and m.thresholds == model.thresholds and m.calibrators is model.calibrators
        assert m.pipeline is not model.pipeline
    te = folds[0][1]
    # probabilities of the toy model saturate at 0 and 1, so compare the raw decision scores
    assert not np.allclose(models[0].pipeline.decision(ds.X[te]), model.pipeline.decision(ds.X[te]), atol=1e-3)


def test_manifest_folds_follow_the_group_split(toy_full):
    cfg, model, ds, folds = toy_full
    man = pd.DataFrame({"group_id": list(ds.groups[:6]) + ["never_seen_a", "never_seen_b"]})
    f = sc.manifest_folds(man, ds, folds)
    expect = {g: i for i, (_, te) in enumerate(folds) for g in np.unique(ds.groups[te])}
    assert f[:6].tolist() == [expect[g] for g in ds.groups[:6]]
    assert set(f[6:]) <= set(range(len(folds)))


# ---- script helper -------------------------------------------------------------------------------------------------------------

def test_false_alarm_breakdown_by_clip_type():
    spec = importlib.util.spec_from_file_location("eval_fa", Path(__file__).resolve().parents[1] / "scripts" / "eval_false_alarms.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    segs = [sc.Segment(0.0, 300.0, "rain", False), sc.Segment(300.0, 600.0, "engine", True)]
    p = np.zeros(1200)
    p[100:112] = 0.9                                         # t = 50 s, in the rain clip
    p[700:712] = 0.9                                         # t = 350 s, in the engine clip
    p[900:912] = 0.9                                         # t = 450 s, in the engine clip
    df, out = m.fa_breakdown([sc.BgStream("esc50_crossfit", _obs(p), 600 / 3600, segs)], sc.params_for(ALERT, 0.5, 0.3), "aircraft")
    assert df["category"].tolist() == ["rain", "engine", "engine"]
    assert out["hard_negative"]["false_alarms"] == 2 and out["other_ambient"]["false_alarms"] == 1
    assert out["hard_negative"]["hours"] == pytest.approx(300 / 3600) and out["hard_negative"]["per_hour"] == pytest.approx(24.0)


def test_own_recordings_are_scored_with_the_saved_model_at_any_sample_rate(toy, tmp_path):
    import copy
    import soundfile as sf
    cfg, model = toy
    cfg = copy.deepcopy(cfg)
    cfg["m6"]["false_alarms"]["own_dir"] = str(tmp_path)
    sf.write(tmp_path / "garden_20261012_1015.wav", noise(20.0, 0.03, 1), SR)
    sf.write(tmp_path / "road_20261012_1100.wav", np.repeat(noise(10.0, 0.03, 2), 2), 2 * SR)     # 32 kHz, 10 s
    (tmp_path / "notes.txt").write_text("not audio")
    spec = importlib.util.spec_from_file_location("eval_fa2", Path(__file__).resolve().parents[1] / "scripts" / "eval_false_alarms.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    out = m.own_streams(cfg, model, "aircraft", 1.0)
    assert [b.source for b in out] == ["own", "own"] and sum(b.hours for b in out) == pytest.approx(30.0 / 3600)
    assert [len(b.obs) for b in out] == [39, 19]
    cfg["m6"]["false_alarms"]["own_dir"] = str(tmp_path / "missing")
    assert m.own_streams(cfg, model, "aircraft", 1.0) == []


def test_threshold_grid_contains_the_extra_value_exactly_and_is_sorted():
    for thr in (0.3017998186492763, 0.30177, 0.3018, 0.2999999, 0.5):
        g = sc.threshold_grid(0.04, 0.96, 0.02, [thr])
        assert thr in g and g == sorted(g) and len(set(g)) == len(g)
        assert np.isclose(pd.Series(g), thr).any()
    g = sc.threshold_grid(0.04, 0.96, 0.02)
    assert g[0] == 0.04 and g[-1] == 0.96 and len(g) == 47
