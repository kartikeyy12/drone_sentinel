"""Range model: closed form, alpha solver, snr_min from a curve, sensitivity table, results.json readers."""
from __future__ import annotations

import copy
import math

import pytest

from sentinel import range_model as rm
from sentinel.config import load_config


def test_closed_form_matches_the_condition_at_the_boundary():
    d = rm.range_m(80.0, 45.0, 10.0)
    assert d == pytest.approx(10 ** ((80 - 45 - 10) / 20))
    assert rm.margin_db(d, 80.0, 45.0, 10.0) == pytest.approx(0.0, abs=1e-9)
    assert rm.margin_db(d * 1.01, 80.0, 45.0, 10.0) < 0 < rm.margin_db(d * 0.99, 80.0, 45.0, 10.0)


def test_six_db_per_doubling_of_distance():
    assert rm.range_m(80.0 + 20 * math.log10(2), 45.0, 10.0) == pytest.approx(2 * rm.range_m(80.0, 45.0, 10.0))


def test_absorption_shortens_the_range_and_solves_the_condition():
    d0 = rm.range_m(80.0, 45.0, 10.0)
    d = rm.range_m(80.0, 45.0, 10.0, alpha_db_per_m=0.01)
    assert 0 < d < d0
    assert rm.margin_db(d, 80.0, 45.0, 10.0, 0.01) == pytest.approx(0.0, abs=1e-6)


def test_not_detectable_even_at_the_reference_distance_gives_none():
    assert rm.range_m(50.0, 45.0, 10.0) is None


def test_reference_distance_is_respected():
    assert rm.range_m(60.0, 50.0, 10.0, d0_m=2.0) == pytest.approx(2.0)


def test_bad_inputs_raise():
    with pytest.raises(ValueError):
        rm.range_m(80, 45, 10, alpha_db_per_m=-0.1)
    with pytest.raises(ValueError):
        rm.range_m(80, 45, 10, d0_m=0.0)


def test_snr_min_needs_every_higher_point_to_pass():
    snr = [20, 10, 5, 0, -5]
    assert rm.snr_min_from_curve(snr, [1.0, 0.95, 0.92, 0.5, 0.95], 0.9) == 5.0       # the lucky -5 dB point does not count
    assert rm.snr_min_from_curve(snr, [1.0, 1.0, 1.0, 1.0, 1.0], 0.9) == -5.0
    assert rm.snr_min_from_curve(snr, [0.5, 1.0, 1.0, 1.0, 1.0], 0.9) is None        # best SNR already misses: not reached
    assert rm.snr_min_from_curve([-5, 0, 5], [0.9, 0.9, 0.9], 0.9) == -5.0             # order of the input does not matter


RES = {"false_alarms": {"at_operating_point": {"by_snr": {"+20": {"event_recall": 0.075}, "+10": {"event_recall": 0.0}}},
                        "at_model_threshold": {"by_snr": {"+20": {"event_recall": 0.275}, "+10": {"event_recall": 0.175}}}},
       "robustness": {"records": [{"tag": "clean_trained", "variant": "none", "kind": "white", "snr_db": 15.0, "recall_aircraft": 0.5},
                                  {"tag": "clean_trained", "variant": "none", "kind": "white", "snr_db": 5.0, "recall_aircraft": 0.2},
                                  {"tag": "clean_trained", "variant": "wiener", "kind": "white", "snr_db": 15.0, "recall_aircraft": 0.9},
                                  {"tag": "clean_trained", "variant": "none", "kind": "clean", "snr_db": None, "recall_aircraft": 0.75}]}}


def test_curves_are_read_from_results_and_variant_none_only():
    c = rm.curves_from_results(RES)
    assert c["m6_event_recall_at_operating_point"]["snr_db"] == [20.0, 10.0]
    assert c["m6_event_recall_at_operating_point"]["recall"] == [0.075, 0.0]
    assert c["m4_window_recall_white"]["recall"] == [0.5, 0.2] and c["m4_window_recall_white"]["unit"] == "window"
    assert "m4_window_recall_pink" not in c


def test_analyse_reports_not_reached_and_never_changes_the_target():
    cfg = load_config()
    a = rm.analyse(RES, cfg)
    assert a["recall_target"] == cfg["m7"]["recall_target"] == 0.9
    assert a["snr_min_db"] is None and not a["snr_min_reached"] and a["range_estimate_m"] is None
    assert any("never reaches" in m for m in a["range_estimate_missing_inputs"])
    assert a["curve_summary"]["m6_event_recall_at_model_threshold"]["snr_min_db_for_target"]["0.25"] == 20.0
    assert len(a["sensitivity"]) == 3 * 7 * 4 and a["simplifications"]


def test_analyse_gives_an_estimate_only_when_every_input_is_set():
    cfg = copy.deepcopy(load_config())
    cfg["m7"]["recall_target"] = 0.05                       # reached by the toy curve
    cfg["m7"]["drone_source_level_db_at_1m"] = 80.0
    a = rm.analyse(RES, cfg)
    assert a["snr_min_db"] == 20.0 and any("ambient" in m for m in a["range_estimate_missing_inputs"]) and a["range_estimate_m"] is None
    cfg["m7"]["ambient_level_db_site"] = 45.0
    a = rm.analyse(RES, cfg)
    assert a["range_estimate_missing_inputs"] == [] and a["range_estimate_m"] == pytest.approx(10 ** ((80 - 45 - 20) / 20))


def test_source_level_is_not_assumed_in_the_default_config():
    m7 = load_config()["m7"]
    assert m7["drone_source_level_db_at_1m"] is None and m7["ambient_level_db_site"] is None


def test_sensitivity_rows_cover_the_grid_and_are_monotonic_in_source_level():
    rows = rm.sensitivity_table([60, 70, 80], [35, 55], [10.0])
    assert len(rows) == 6
    r35 = [r["range_m"] for r in rows if r["ambient_level_db"] == 35]
    assert all(r35[i] < r35[i + 1] for i in range(2))
    assert math.isclose(r35[1] / r35[0], 10 ** 0.5)
