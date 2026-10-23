"""Windowing helpers."""
from __future__ import annotations

import numpy as np


def n_windows(duration_s: float, window_s: float = 1.0, hop_s: float = 0.5) -> int:
    """Number of full windows that fit in `duration_s` (no padding)."""
    if duration_s < window_s:
        return 0
    # round() guards against float error, e.g. (5.0 - 1.0) / 0.5 = 8.000000000000002 style cases
    return int(round((duration_s - window_s) / hop_s, 9) // 1) + 1


def window_signal(x: np.ndarray, sr: int, window_s: float = 1.0, hop_s: float = 0.5
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Cut `x` into full windows. Returns (windows [n, win_samples] view, start times [n] in s).

    The last partial window is dropped (no padding), so clips shorter than one window give n = 0.
    """
    win, hop = int(round(window_s * sr)), int(round(hop_s * sr))
    n = 0 if len(x) < win else (len(x) - win) // hop + 1
    if n == 0:
        return np.zeros((0, win), dtype=x.dtype), np.zeros(0, dtype=np.float32)
    idx = np.arange(win)[None, :] + hop * np.arange(n)[:, None]
    return x[idx], (hop * np.arange(n) / sr).astype(np.float32)
