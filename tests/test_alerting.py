"""Alert state machine and its pure helpers. Windows are 1.0 s long with a 0.5 s hop: window k spans [0.5k, 0.5k + 1.0]."""
from __future__ import annotations

import numpy as np
import pytest

from sentinel import alerting as al
from sentinel.alerting import AlertParams, Obs, TrackerState, detect_events, flush, step

HOP, WIN = 0.5, 1.0


def series(p, power=0.0, floor=None, alt=()):
    return [Obs(i * HOP, i * HOP + WIN, float(x), power, floor, alt) for i, x in enumerate(p)]


def params(**kw):
    base = dict(t_on=0.5, t_off=0.3, ema_alpha=1.0, n_on=3, m_window=5, k_off=4, min_event_s=2.0, merge_gap_s=3.0)
    return AlertParams(**{**base, **kw})


def run_steps(pr, obs):
    st, out = TrackerState(label="aircraft"), []
    for o in obs:
        st, ev = step(pr, st, o)
        out.append(ev)
    return st, out


# ---- helpers -----------------------------------------------------------------------------------------------------------

def test_ema_update_step_response_and_first_value():
    assert al.ema_update(None, 0.7, 0.3) == 0.7
    s, out = None, []
    for x in (0, 1, 1, 1):
        s = al.ema_update(s, x, 0.5)
        out.append(s)
    assert out == pytest.approx([0.0, 0.5, 0.75, 0.875])
    assert al.ema_update(0.2, 0.9, 1.0) == 0.9  # alpha 1 means no smoothing


def test_frame_powers_of_a_sine_and_partial_frame_dropped():
    x = 0.5 * np.sin(2 * np.pi * 440 * np.arange(1000) / 16000)
    p = al.frame_powers(x, 320)
    assert len(p) == 3 and p == pytest.approx(0.5 * 0.25, rel=0.05)
    assert len(al.frame_powers(x[:100], 320)) == 0


def test_noise_floor_percentile_and_min_frames():
    p = np.arange(1.0, 101.0)
    assert al.noise_floor(p, 10.0) == pytest.approx(np.percentile(p, 10))
    assert al.noise_floor(p[:5], 10.0, min_frames=10) is None
    assert al.noise_floor(np.zeros(0)) is None


def test_snr_db_power_subtraction_and_guards():
    assert al.snr_db(2.0, 1.0) == pytest.approx(0.0)          # P = 2N: signal equals noise
    assert al.snr_db(11.0, 1.0) == pytest.approx(10.0)
    assert al.snr_db(101.0, 1.0) == pytest.approx(20.0)
    assert al.snr_db(1.0, 1.0) == -20.0 and al.snr_db(0.5, 1.0) == -20.0   # at or below the floor: lowest value
    assert al.snr_db(5.0, None) is None
    assert al.snr_db(5.0, 1e-12) is None                       # digital silence floor: no meaningful ratio
    assert al.snr_db(float("nan"), 1.0) is None


@pytest.mark.parametrize("true_db", [-3.0, 0.0, 6.0, 12.0, 20.0])
def test_snr_estimate_recovers_a_known_snr_on_synthetic_audio(true_db):
    rng = np.random.default_rng(0)
    fl = 320
    noise = rng.standard_normal(16000 * 60) * 0.01
    sig = np.sin(2 * np.pi * 800 * np.arange(16000 * 10) / 16000) * np.sqrt(2) * 0.01 * 10 ** (true_db / 20)
    floor = al.noise_floor(al.frame_powers(noise, fl), 10.0)
    ev = noise[: len(sig)] + sig
    est = al.snr_db(float(np.mean(al.frame_powers(ev, fl))), floor)
    # the 10th percentile of frame power of Gaussian noise sits below the mean power, which biases the estimate up a little
    assert est == pytest.approx(true_db, abs=2.0)


def test_window_label_rules():
    thr = {"aircraft": 0.3, "drone": 0.5}
    assert al.window_label({"aircraft": 0.1, "drone": 0.1}, thr) == "ambient"
    assert al.window_label({"aircraft": 0.4, "drone": 0.1}, thr) == "aircraft"
    assert al.window_label({"aircraft": 0.4, "drone": 0.6}, thr) == "drone"           # highest probability among those over threshold
    assert al.window_label({"aircraft": 0.9, "drone": 0.9}, thr, gated=True) == "silence"
    # reject option: ambient counts as 1 - max target probability
    assert al.window_label({"aircraft": 0.5, "drone": 0.1}, thr, reject_below=0.6) == "unknown"
    assert al.window_label({"aircraft": 0.9, "drone": 0.1}, thr, reject_below=0.6) == "aircraft"
    assert al.window_label({"aircraft": 0.05, "drone": 0.1}, thr, reject_below=0.6) == "ambient"
    assert al.window_label({"aircraft": 0.5, "drone": 0.1}, thr, reject_below=None) == "aircraft"


def test_params_validation_and_from_cfg():
    for bad in (dict(t_off=0.6), dict(t_on=1.5), dict(ema_alpha=0.0), dict(n_on=6), dict(n_on=0), dict(k_off=0),
                dict(min_event_s=-1), dict(merge_gap_s=-1)):
        with pytest.raises(ValueError):
            params(**bad)
    cfg = {"ema_alpha": 0.5, "t_on": None, "t_off": None, "t_off_ratio": 0.6, "n_on": 3, "m_window": 5, "k_off": 4,
           "min_event_s": 2.0, "merge_gap_s": 3.0}
    p = AlertParams.from_cfg(cfg, default_t_on=0.3)
    assert p.t_on == 0.3 and p.t_off == pytest.approx(0.18)
    p2 = AlertParams.from_cfg({**cfg, "t_on": 0.8, "t_off": 0.4}, default_t_on=0.3)
    assert (p2.t_on, p2.t_off) == (0.8, 0.4)


# ---- onset, end, hysteresis ------------------------------------------------------------------------------------------------

def test_quiet_stream_gives_no_events():
    assert detect_events(params(), series([0.05] * 100), "aircraft") == []


def test_sustained_signal_gives_one_back_dated_event():
    p = [0] * 4 + [0.9] * 10 + [0] * 8                     # windows 4..13 are above t_on
    ev = detect_events(params(), series(p), "aircraft")
    assert len(ev) == 1
    e = ev[0]
    assert e.t_start == pytest.approx(2.0)                  # window 4 starts at 2.0: back-dated to the first above window
    assert e.t_alert == pytest.approx(4.0)                  # the third above window (6) ended at 3.0 + 1.0
    assert e.t_end == pytest.approx(7.5)                    # window 13 ends at 6.5 + 1.0
    assert e.duration_s == pytest.approx(5.5) and e.n_windows == 10
    assert e.peak_prob == pytest.approx(0.9) and e.mean_prob == pytest.approx(0.9) and e.label == "aircraft"
    assert e.t_start <= e.t_alert <= e.t_end


def test_n_of_m_needs_n_above_threshold_windows_within_the_last_m():
    pr = params()
    assert detect_events(pr, series([0, 0.9, 0.9, 0, 0, 0, 0, 0, 0, 0]), "x") == []          # only 2 windows
    assert detect_events(pr, series([0.9, 0.9, 0, 0, 0, 0.9, 0.9, 0, 0, 0, 0, 0, 0]), "x") == []   # 2 + 2, never 3 in 5
    e = detect_events(pr, series([0.9, 0, 0.9, 0, 0.9] + [0] * 8), "x")                       # 3 of the last 5 at window 4
    assert len(e) == 1 and e[0].t_start == 0.0 and e[0].t_alert == pytest.approx(3.0) and e[0].t_end == pytest.approx(3.0)


def test_alert_raised_exactly_at_the_nth_window():
    st, outs = run_steps(params(), series([0.9] * 3))
    assert st.active                                         # ON after the third window, not the second
    st2, _ = run_steps(params(), series([0.9] * 2))
    assert not st2.active


def test_hysteresis_keeps_the_event_open_between_t_off_and_t_on():
    p = [0.9] * 5 + [0.4] * 10 + [0.9] * 5 + [0] * 8       # 0.4 is above t_off = 0.3 and below t_on = 0.5
    ev = detect_events(params(), series(p), "x")
    assert len(ev) == 1 and ev[0].t_start == 0.0 and ev[0].t_end == pytest.approx(19 * HOP + WIN)


def test_off_needs_k_consecutive_windows_below_t_off():
    one = detect_events(params(k_off=4), series([0.9] * 5 + [0.0] * 3 + [0.9] * 5 + [0] * 8), "x")
    assert len(one) == 1                                     # a 3-window dip (K - 1) does not close the event
    two = detect_events(params(k_off=4, merge_gap_s=0.0), series([0.9] * 5 + [0.0] * 4 + [0.9] * 5 + [0] * 8), "x")
    assert len(two) == 2                                     # a 4-window dip closes it (no merging here)
    assert two[0].t_end == pytest.approx(4 * HOP + WIN) and two[1].t_start == pytest.approx(9 * HOP)


def test_no_retrigger_from_the_evidence_of_the_event_that_just_closed():
    # K = 1 closes at the first low window; the five above-threshold windows before it must not open a second event
    ev = detect_events(params(k_off=1, merge_gap_s=0.0), series([0.9] * 5 + [0.0] * 10), "x")
    assert len(ev) == 1


def test_ema_delays_the_onset():
    pr_fast = params(t_on=0.6, ema_alpha=1.0, n_on=1, m_window=1)
    pr_slow = params(t_on=0.6, ema_alpha=0.5, n_on=1, m_window=1)
    p = [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0] + [0.0] * 10
    a = detect_events(pr_fast, series(p), "x")[0]
    b = detect_events(pr_slow, series(p), "x")[0]
    assert a.t_start == pytest.approx(1.0) and a.t_alert == pytest.approx(2.0)   # window 2 is above t_on at once
    # smoothed values 0, 0, 0.5, 0.75: the first one at or above 0.6 is window 3 (start 1.5, end 2.5)
    assert b.t_start == pytest.approx(1.5) and b.t_alert == pytest.approx(2.5)


# ---- minimum duration and merging -------------------------------------------------------------------------------------------

def test_events_shorter_than_the_minimum_duration_are_dropped():
    pr = params(n_on=1, m_window=1, k_off=1, merge_gap_s=0.0, min_event_s=2.0)
    st, _ = run_steps(pr, series([0.9, 0.9, 0, 0, 0]))
    assert st.n_dropped_short == 1                           # two windows span 1.5 s
    assert detect_events(pr, series([0.9, 0.9, 0.9, 0, 0, 0]), "x")[0].duration_s == pytest.approx(2.0)   # exactly 2.0 s is kept


@pytest.mark.parametrize("zeros,merged", [(3, True), (7, True), (8, False), (20, False)])
def test_gap_merging_boundary(zeros, merged):
    # event 1 ends at 3.0 (window 4); event 2 starts at (5 + zeros) * 0.5; the gap is 0.5 * zeros - 0.5 seconds
    p = [0.9] * 5 + [0] * zeros + [0.9] * 5 + [0] * 12
    ev = detect_events(params(merge_gap_s=3.0), series(p), "x")
    assert (len(ev) == 1) == merged
    if merged:
        assert ev[0].t_start == 0.0 and ev[0].t_end == pytest.approx((5 + zeros + 4) * HOP + WIN)
    else:
        assert ev[0].t_end == pytest.approx(4 * HOP + WIN) and ev[1].t_start == pytest.approx((5 + zeros) * HOP)


def test_merged_event_statistics_include_the_gap_windows():
    p = [0.9] * 5 + [0.0] * 5 + [0.9] * 5 + [0] * 12        # a 5-window gap closes the first event, then it merges
    e = detect_events(params(), series(p), "x")[0]
    assert e.n_windows == 15 and e.peak_prob == pytest.approx(0.9) and e.mean_prob == pytest.approx(0.9 * 10 / 15)


def test_a_closed_event_is_held_back_until_it_can_no_longer_merge():
    p = [0] * 4 + [0.9] * 10 + [0] * 20
    st, outs = TrackerState(label="x"), []
    pr = params()
    emitted_at = None
    for i, o in enumerate(series(p)):
        st, ev = step(pr, st, o)
        if ev and emitted_at is None:
            emitted_at = i
    # event ends at 7.5; it is released when the window start exceeds 7.5 + 3.0, i.e. window 22 (start 11.0)
    assert emitted_at == 22


def test_merge_gap_zero_emits_at_close():
    pr = params(merge_gap_s=0.0)
    st, outs = run_steps(pr, series([0.9] * 5 + [0.0] * 4))
    assert len(outs[-1]) == 1 and outs[-1][0].t_end == pytest.approx(4 * HOP + WIN)


# ---- flush ----------------------------------------------------------------------------------------------------------------------

def test_flush_closes_the_open_event_at_its_last_good_window():
    pr = params()
    st, outs = run_steps(pr, series([0.9] * 6 + [0.0] * 2))   # still open: 2 low windows < K
    assert st.active and not any(outs)
    st2, ev = flush(pr, st)
    assert len(ev) == 1 and ev[0].t_end == pytest.approx(5 * HOP + WIN) and ev[0].n_windows == 6
    assert not st2.active and flush(pr, st2)[1] == []


def test_flush_releases_a_held_back_event_and_drops_a_too_short_open_one():
    pr = params()
    st, _ = run_steps(pr, series([0.9] * 5 + [0.0] * 4))      # closed, waiting for a possible merge
    assert st.pending is not None
    assert len(flush(pr, st)[1]) == 1
    pr1 = params(n_on=1, m_window=1)
    st, _ = run_steps(pr1, series([0.9, 0.9]))                # open but only 1.5 s long
    st, ev = flush(pr1, st)
    assert ev == [] and st.n_dropped_short == 1


# ---- purity and invariants -----------------------------------------------------------------------------------------------------

def test_step_is_pure_and_deterministic():
    pr = params()
    obs = series([0.9] * 8)
    st = TrackerState(label="x")
    snapshots = []
    for o in obs:
        before = st
        st1, e1 = step(pr, st, o)
        st2, e2 = step(pr, st, o)
        assert st1 == st2 and e1 == e2 and st is before
        snapshots.append(before)
        st = st1
    with pytest.raises(Exception):
        st.ema = 0.0  # frozen


def test_detect_events_equals_streaming_with_a_final_flush():
    rng = np.random.default_rng(3)
    p = np.clip(rng.random(300) ** 2, 0, 1)
    pr = params(ema_alpha=0.6)
    batch = detect_events(pr, series(p), "x")
    st, out = TrackerState(label="x"), []
    for o in series(p):
        st, ev = step(pr, st, o)
        out += ev
    assert batch == out + flush(pr, st)[1]


def test_random_series_keep_the_event_invariants():
    rng = np.random.default_rng(0)
    for trial in range(300):
        n = int(rng.integers(20, 400))
        p = np.clip(rng.random(n) ** rng.uniform(0.5, 4) + rng.normal(0, 0.05, n), 0, 1)
        t_on = float(rng.uniform(0.3, 0.9))
        m = int(rng.integers(1, 8))
        pr = AlertParams(t_on=t_on, t_off=float(rng.uniform(0.05, t_on - 0.02)), ema_alpha=float(rng.uniform(0.2, 1.0)),
                         n_on=int(rng.integers(1, m + 1)), m_window=m, k_off=int(rng.integers(1, 6)),
                         min_event_s=float(rng.uniform(0, 4)), merge_gap_s=float(rng.uniform(0, 5)))
        ev = detect_events(pr, series(p), "x")
        for a, b in zip(ev, ev[1:]):
            assert b.t_start - a.t_end > pr.merge_gap_s - 1e-6      # anything closer than the merge gap was merged
        for e in ev:
            assert e.duration_s >= pr.min_event_s - 1e-6
            assert e.t_start <= e.t_alert <= e.t_end + 1e-9
            assert 0 <= e.mean_prob <= e.peak_prob <= 1
            assert e.t_end <= (n - 1) * HOP + WIN + 1e-9 and e.t_start >= 0


# ---- event statistics -------------------------------------------------------------------------------------------------------------

def test_event_top_alt_class_snr_and_probabilities():
    alt = (("drone", 0.1), ("ambient", 0.4))
    obs = [Obs(i * HOP, i * HOP + WIN, 0.9, power=11.0, floor=1.0, alt=alt) for i in range(6)]
    tail = [Obs((6 + i) * HOP, (6 + i) * HOP + WIN, 0.0) for i in range(8)]
    ev = detect_events(params(), obs + tail, "aircraft")
    assert len(ev) == 1 and ev[0].top_alt_class == "ambient"
    assert ev[0].snr_est_db == pytest.approx(10.0)
    no_floor = detect_events(params(), series([0.9] * 6 + [0] * 8), "aircraft")
    assert no_floor[0].snr_est_db is None and no_floor[0].top_alt_class == "ambient"   # defaults when nothing is known


def test_event_snr_uses_the_median_floor_and_mean_power():
    obs = tuple(Obs(i * HOP, i * HOP + WIN, 0.9, power=p, floor=f) for i, (p, f) in enumerate([(6, 1), (11, 1), (21, 100)]))
    # mean power 12.67; median floor 1.0
    assert al.event_snr(obs) == pytest.approx(10 * np.log10(np.mean([6, 11, 21]) / 1.0 - 1))
    assert al.event_snr((Obs(0, 1, 0.9, power=3.0, floor=None),)) is None
