"""Manifest logic on a synthetic ESC-50-shaped tree (no real data needed)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from sentinel.manifest import COLUMNS, dads_rows, esc50_rows, own_rows, summarize, validate_manifest

CFG = {"aircraft": ["airplane"], "ambient": ["rain", "dog"], "hard_negative": ["engine"]}


def _wav(path, dur=5.0, sr=8000):
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.zeros(int(dur * sr), dtype=np.float32), sr)


@pytest.fixture
def esc(tmp_path):
    meta = pd.DataFrame({
        "filename": ["1-1-A-0.wav", "1-1-B-0.wav", "2-2-A-1.wav", "3-3-A-2.wav", "4-4-A-3.wav", "5-5-A-4.wav"],
        "fold": [1, 1, 2, 3, 4, 5], "target": [0, 0, 1, 2, 3, 4],
        "category": ["airplane", "airplane", "rain", "engine", "dog", "cat"],
        "esc10": False, "src_file": [1, 1, 2, 3, 4, 5], "take": list("ABAAAA")})
    (tmp_path / "meta").mkdir()
    meta.to_csv(tmp_path / "meta" / "esc50.csv", index=False)
    for f in meta["filename"]:
        _wav(tmp_path / "audio" / f)
    return tmp_path


def test_esc50_mapping(esc):
    df = pd.DataFrame(esc50_rows(esc, CFG), columns=COLUMNS)
    assert len(df) == 5  # 'cat' is not configured
    by = df.set_index(df["path"].str.split("/").str[-1])
    assert by.loc["1-1-A-0.wav", "label"] == "aircraft"
    assert by.loc["2-2-A-1.wav", "label"] == "ambient" and not by.loc["2-2-A-1.wav", "hard_negative"]
    assert by.loc["3-3-A-2.wav", "label"] == "ambient" and by.loc["3-3-A-2.wav", "hard_negative"]
    assert (df["sr"] == 8000).all() and (df["duration_s"] == 5.0).all()


def test_esc50_same_source_shares_group(esc):
    df = pd.DataFrame(esc50_rows(esc, CFG), columns=COLUMNS)
    air = df[df["category"] == "airplane"]
    assert air["group_id"].nunique() == 1 and len(air) == 2


def test_esc50_missing_audio(esc):
    (esc / "audio" / "4-4-A-3.wav").unlink()
    with pytest.raises(FileNotFoundError):
        esc50_rows(esc, CFG)
    assert len(esc50_rows(esc, CFG, allow_missing=True)) == 4


def test_esc50_unknown_category(esc):
    with pytest.raises(ValueError):
        esc50_rows(esc, {**CFG, "ambient": ["rain", "not_a_class"]})


def test_own_naming_and_bench_skipped(tmp_path):
    _wav(tmp_path / "hostel_20261012_1015.wav", 3.0)
    _wav(tmp_path / "cafe_corner_20261012_1100.wav", 3.0)
    _wav(tmp_path / "bench_1m.wav", 3.0)
    rows = own_rows(tmp_path)
    assert sorted(r["group_id"] for r in rows) == ["own_cafe_corner_20261012_1100", "own_hostel_20261012_1015"]
    assert all(r["label"] == "ambient" and r["dataset"] == "own" for r in rows)


def test_dads_requires_layout(tmp_path):
    with pytest.raises(ValueError, match="not set"):
        dads_rows(tmp_path, {"audio_glob": None})


def test_dads_regex_rows(tmp_path):
    _wav(tmp_path / "drone" / "rec7_seg0.wav", 1.0)
    _wav(tmp_path / "drone" / "rec7_seg1.wav", 1.0)
    _wav(tmp_path / "other" / "city3_seg0.wav", 1.0)
    cfg = {"audio_glob": "**/*.wav", "drone_path_regex": r"^drone/", "nondrone_path_regex": r"^other/",
           "group_regex": r"/([a-z]+\d+)_seg\d+\.wav$"}
    df = pd.DataFrame(dads_rows(tmp_path, cfg), columns=COLUMNS)
    assert sorted(df["group_id"].unique()) == ["dads_city3", "dads_rec7"]
    assert df[df["group_id"] == "dads_rec7"]["label"].unique().tolist() == ["drone"]
    bad = {**cfg, "nondrone_path_regex": r"^drone/|^other/"}
    with pytest.raises(ValueError, match="both"):
        dads_rows(tmp_path, bad)


def test_validate_catches_problems(esc):
    df = pd.DataFrame(esc50_rows(esc, CFG), columns=COLUMNS)
    assert validate_manifest(df) == []
    bad = df.copy()
    bad.loc[bad.index[0], "group_id"] = bad.loc[bad.index[2], "group_id"]  # same group, two labels
    assert any("more than one label" in p for p in validate_manifest(bad))
    dup = pd.concat([df, df.iloc[:1]])
    assert any("duplicate" in p for p in validate_manifest(dup))


def test_summarize_counts(esc):
    df = pd.DataFrame(esc50_rows(esc, CFG), columns=COLUMNS)
    s = summarize(df, 1.0, 0.5)
    assert s["by_class"]["aircraft"]["files"] == 2 and s["by_class"]["aircraft"]["groups"] == 1
    assert s["by_class"]["aircraft"]["windows"] == 2 * 9  # 5 s clip -> 9 windows
    assert s["by_dataset_bucket"]["esc50/ambient_hard_negative"]["files"] == 1


def test_esc50_missing_meta_gives_hint(tmp_path):
    with pytest.raises(FileNotFoundError, match="download_data.py --only esc50"):
        esc50_rows(tmp_path, CFG)
