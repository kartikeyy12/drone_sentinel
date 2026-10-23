"""Audio loading and filtering: resample, DC removal, Butterworth band-pass (offline and streaming)."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from scipy import signal


def resample(x: np.ndarray, fs: int, sr: int) -> np.ndarray:
    """Polyphase resample of a mono signal from fs to sr (identity if equal)."""
    if int(fs) == int(sr):
        return x.astype(np.float32, copy=False)
    g = math.gcd(int(fs), int(sr))
    return signal.resample_poly(x, int(sr) // g, int(fs) // g).astype(np.float32)


def load_audio(path: str | Path, sr: int) -> np.ndarray:
    """Read any soundfile-readable file, average channels to mono, resample to sr. No amplitude normalisation."""
    x, fs = sf.read(str(path), dtype="float32", always_2d=True)
    return resample(x.mean(axis=1), fs, sr)


def remove_dc(x: np.ndarray) -> np.ndarray:
    """Subtract the mean (offline). The streaming path relies on the band-pass low cut instead."""
    return x - x.mean() if len(x) else x


def bandpass_sos(sr: int, low_hz: float, high_hz: float, order: int) -> np.ndarray:
    """Butterworth band-pass as second-order sections."""
    return signal.butter(order, [low_hz, high_hz], btype="bandpass", fs=sr, output="sos")


def bandpass_offline(x: np.ndarray, sos: np.ndarray) -> np.ndarray:
    """Zero-phase filtering (sosfiltfilt). Padding shrinks automatically for very short inputs."""
    if len(x) < 4:
        return x.copy()
    default_pad = 3 * (2 * len(sos) + 1 - min(int((sos[:, 2] == 0).sum()), int((sos[:, 5] == 0).sum())))
    return signal.sosfiltfilt(sos, x, padlen=min(default_pad, len(x) - 1)).astype(np.float32)


class StreamingBandpass:
    """Causal band-pass with filter state carried across chunks (used by the live engine).

    Starts from zero state, i.e. from silence. Chunked output equals one-shot sosfilt output exactly.
    """

    def __init__(self, sos: np.ndarray) -> None:
        self.sos = sos
        self.zi = np.zeros((len(sos), 2))

    def process(self, chunk: np.ndarray) -> np.ndarray:
        y, self.zi = signal.sosfilt(self.sos, chunk, zi=self.zi)
        return y.astype(np.float32)

    def reset(self) -> None:
        self.zi = np.zeros((len(self.sos), 2))


def preprocess_clip(x: np.ndarray, sr: int, cfg: dict[str, Any]) -> np.ndarray:
    """Offline preprocessing of a whole clip: DC removal then zero-phase band-pass."""
    bp = cfg["preprocess"]["bandpass"]
    sos = bandpass_sos(sr, bp["low_hz"], bp["high_hz"], bp["order"])
    return bandpass_offline(remove_dc(x), sos)
