"""Build and validate data/manifest.csv from the supported datasets.

Columns (required): path, dataset, label, group_id, duration_s, sr, hard_negative
Extra column:       category (original category/class name, used for per-category reporting)

`path` is relative to the project root. `group_id` identifies the source recording and is the
unit of grouped splitting: windows from one group must never be in both train and test.
"""
from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any

import pandas as pd
import soundfile as sf

from .config import PROJECT_ROOT
from .windowing import n_windows

log = logging.getLogger(__name__)

COLUMNS = ["path", "dataset", "label", "group_id", "duration_s", "sr", "hard_negative", "category"]
LABELS = {"drone", "aircraft", "ambient"}
OWN_NAME = re.compile(r"^(?P<place>.+)_(?P<date>\d{8})_(?P<time>\d{4})$")


def audio_info(path: Path) -> tuple[float, int]:
    """Return (duration_s, sample_rate) from the file header without decoding."""
    info = sf.info(str(path))
    return float(info.duration), int(info.samplerate)


def _rel(path: Path) -> str:
    # abspath (not resolve) so a symlinked data/raw/<name> keeps its project-relative path
    p = Path(os.path.abspath(path))
    try:
        return p.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return p.as_posix()


def _row(path: Path, dataset: str, label: str, group_id: str, hard_negative: bool,
         category: str) -> dict[str, Any]:
    dur, sr = audio_info(path)
    return {"path": _rel(path), "dataset": dataset, "label": label, "group_id": group_id,
            "duration_s": round(dur, 3), "sr": sr, "hard_negative": bool(hard_negative),
            "category": category}


def esc50_rows(esc_root: Path, ds_cfg: dict[str, Any], allow_missing: bool = False
               ) -> list[dict[str, Any]]:
    """Rows for the configured ESC-50 categories.

    group_id = 'esc50_<src_file>': src_file is the Freesound source clip id, so clips cut from the
    same source recording share a group (1524 unique sources across the 2000 clips, from meta CSV).
    """
    if not (esc_root / "meta" / "esc50.csv").exists():
        raise FileNotFoundError(f"ESC-50 not found at {esc_root}. Run: python scripts/download_data.py --only esc50")
    meta = pd.read_csv(esc_root / "meta" / "esc50.csv")
    cat_label = {c: "aircraft" for c in ds_cfg["aircraft"]}
    cat_label.update({c: "ambient" for c in ds_cfg["ambient"]})
    hard = set(ds_cfg["hard_negative"])
    cat_label.update({c: "ambient" for c in hard})
    unknown = [c for c in cat_label if c not in set(meta["category"])]
    if unknown:
        raise ValueError(f"categories not in esc50.csv: {unknown}")
    rows: list[dict[str, Any]] = []
    missing = 0
    for r in meta[meta["category"].isin(cat_label)].itertuples():
        wav = esc_root / "audio" / r.filename
        if not wav.exists():
            missing += 1
            continue
        rows.append(_row(wav, "esc50", cat_label[r.category], f"esc50_{r.src_file}",
                         r.category in hard, r.category))
    if missing:
        msg = f"{missing} ESC-50 audio files listed in meta are missing under {esc_root / 'audio'}"
        if not allow_missing:
            raise FileNotFoundError(msg)
        log.warning(msg)
    return rows


def own_rows(own_dir: Path) -> list[dict[str, Any]]:
    """Rows for data/own/<place>_<yyyymmdd>_<hhmm>.wav (label ambient, group = file).

    Files that do not match the naming scheme are skipped with a warning, so a drone clip
    such as bench_1m.wav can never be mislabelled as no-drone ambient.
    """
    rows: list[dict[str, Any]] = []
    for wav in sorted(own_dir.glob("*.wav")):
        if not OWN_NAME.match(wav.stem):
            log.warning("skipping %s: name is not <place>_<yyyymmdd>_<hhmm>.wav", wav.name)
            continue
        rows.append(_row(wav, "own", "ambient", f"own_{wav.stem}", False, OWN_NAME.match(wav.stem)["place"]))
    return rows


def dads_rows(dads_root: Path, ds_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Rows for DADS using the regexes in the config (layout must be filled in after inspection)."""
    need = ["audio_glob", "drone_path_regex", "nondrone_path_regex", "group_regex"]
    missing = [k for k in need if not ds_cfg.get(k)]
    if missing:
        raise ValueError(
            f"datasets.dads.{missing} not set in configs/default.yaml. Run scripts/inspect_dads.py, "
            "read docs/data_notes.md and fill them in (the DADS layout is not assumed).")
    drone_re, non_re = re.compile(ds_cfg["drone_path_regex"]), re.compile(ds_cfg["nondrone_path_regex"])
    group_re = re.compile(ds_cfg["group_regex"])
    rows: list[dict[str, Any]] = []
    for wav in sorted(dads_root.glob(ds_cfg["audio_glob"])):
        rel = wav.relative_to(dads_root).as_posix()
        d, n = bool(drone_re.search(rel)), bool(non_re.search(rel))
        if d == n:
            raise ValueError(f"DADS path matches {'both' if d else 'neither'} label regex: {rel}")
        g = group_re.search(rel)
        if not g:
            raise ValueError(f"DADS group_regex does not match: {rel}")
        label = "drone" if d else "ambient"
        rows.append(_row(wav, "dads", label, f"dads_{g.group(1)}", False, label))
    return rows


def validate_manifest(df: pd.DataFrame) -> list[str]:
    """Return a list of problems (empty if the manifest is sound)."""
    problems: list[str] = []
    if list(df.columns) != COLUMNS:
        problems.append(f"columns are {list(df.columns)}, expected {COLUMNS}")
        return problems
    if df["path"].duplicated().any():
        problems.append(f"{int(df['path'].duplicated().sum())} duplicate paths")
    bad = set(df["label"]) - LABELS
    if bad:
        problems.append(f"unknown labels {bad}")
    if df["group_id"].isna().any() or (df["group_id"].astype(str).str.len() == 0).any():
        problems.append("empty group_id")
    if (df["duration_s"] <= 0).any():
        problems.append("non-positive duration")
    multi = df.groupby("group_id")["label"].nunique()
    if (multi > 1).any():
        problems.append(f"groups with more than one label: {list(multi[multi > 1].index[:5])}")
    multi_ds = df.groupby("group_id")["dataset"].nunique()
    if (multi_ds > 1).any():
        problems.append(f"groups spanning datasets: {list(multi_ds[multi_ds > 1].index[:5])}")
    if ((df["label"] == "drone") & df["hard_negative"]).any():
        problems.append("hard_negative set on a drone row")
    return problems


def summarize(df: pd.DataFrame, window_s: float, hop_s: float) -> dict[str, Any]:
    """Counts, minutes, window counts and group counts per dataset/class (for results.json)."""
    d = df.copy()
    d["n_windows"] = [n_windows(x, window_s, hop_s) for x in d["duration_s"]]
    d["bucket"] = d["label"].where(~d["hard_negative"], "ambient_hard_negative")
    out: dict[str, Any] = {"n_files": int(len(d)), "n_groups": int(d["group_id"].nunique()),
                           "window_s": window_s, "hop_s": hop_s, "by_dataset_bucket": {}}
    for (ds, b), g in d.groupby(["dataset", "bucket"]):
        out["by_dataset_bucket"][f"{ds}/{b}"] = {
            "files": int(len(g)), "groups": int(g["group_id"].nunique()),
            "minutes": round(float(g["duration_s"].sum()) / 60.0, 2),
            "windows": int(g["n_windows"].sum()), "sample_rates": sorted(int(s) for s in g["sr"].unique())}
    out["by_class"] = {lab: {"files": int(len(g)), "groups": int(g["group_id"].nunique()),
                             "minutes": round(float(g["duration_s"].sum()) / 60.0, 2),
                             "windows": int(g["n_windows"].sum())} for lab, g in d.groupby("label")}
    return out
