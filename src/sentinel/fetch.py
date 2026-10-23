"""Resumable per-file ESC-50 download from raw.githubusercontent.com.

Each file is fetched on its own with retries and backoff and checked after download, so a flaky
connection only costs the file in flight and a re-run skips everything already valid. TLS
verification stays on.
"""
from __future__ import annotations

import logging
import shutil
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import soundfile as sf
from tqdm import tqdm

log = logging.getLogger(__name__)
RAW_BASE = "https://raw.githubusercontent.com"
ESC50_CLIP_SECONDS = 5.0  # per the ESC-50 README: all clips are 5 s
META_FILES = ("meta/esc50.csv", "LICENSE", "README.md")


def count_audio(dest: Path) -> int:
    """Number of .wav files under dest/audio."""
    return sum(1 for _ in (dest / "audio").glob("*.wav")) if (dest / "audio").exists() else 0


def _valid_clip(path: Path) -> bool:
    try:
        return abs(sf.info(str(path)).duration - ESC50_CLIP_SECONDS) < 0.01
    except Exception:  # noqa: BLE001
        return False


def _download(url: str, out: Path, validate, retries: int, backoff_s: float, timeout_s: float) -> None:
    """Download url to out (via .part), validating before the final rename. Raises RuntimeError."""
    out.parent.mkdir(parents=True, exist_ok=True)
    part = out.with_name(out.name + ".part")
    last: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=timeout_s) as r, open(part, "wb") as f:
                shutil.copyfileobj(r, f)
            if validate is not None and not validate(part):
                raise ValueError("downloaded file failed validation")
            part.replace(out)
            return
        except Exception as e:  # noqa: BLE001
            last = e
            part.unlink(missing_ok=True)
            time.sleep(backoff_s * (2 ** attempt))
    raise RuntimeError(f"{url}: {last}")


def fetch_esc50(repo: str, dest: Path, folds: tuple[int, ...] = (1, 2, 3, 4, 5), workers: int = 6,
                base: str = RAW_BASE, retries: int = 6, backoff_s: float = 1.0, timeout_s: float = 60.0) -> int:
    """Download ESC-50 metadata plus the audio of `folds` into dest. Returns the wav count in dest/audio.

    Re-running resumes: files that already exist and are valid 5 s clips are skipped.
    Raises RuntimeError listing how many files failed after retries.
    """
    def url(path: str) -> str:
        return f"{base}/{repo}/master/{path}"

    for rel in META_FILES:
        out = dest / rel
        if not out.exists():
            _download(url(rel), out, None, retries, backoff_s, timeout_s)
    meta = pd.read_csv(dest / "meta" / "esc50.csv")
    names = meta.loc[meta["fold"].isin(folds), "filename"].tolist()
    todo = [n for n in names if not _valid_clip(dest / "audio" / n)]
    log.info("ESC-50: %d files wanted, %d already valid, %d to download", len(names), len(names) - len(todo), len(todo))
    failures: list[str] = []

    def one(name: str) -> None:
        try:
            _download(url(f"audio/{name}"), dest / "audio" / name, _valid_clip, retries, backoff_s, timeout_s)
        except RuntimeError as e:
            failures.append(str(e))

    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(tqdm(ex.map(one, todo), total=len(todo), desc="esc50 audio", unit="file"))
    if failures:
        raise RuntimeError(f"{len(failures)} of {len(todo)} files failed after {retries} tries (first: {failures[0]}). "
                           "Re-run to resume.")
    return count_audio(dest)
