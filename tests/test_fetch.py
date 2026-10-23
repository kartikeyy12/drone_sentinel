"""fetch_esc50 against a local HTTP server that mimics raw.githubusercontent.com."""
from __future__ import annotations

import functools
import http.server
import threading

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from sentinel.fetch import count_audio, fetch_esc50


def _wav(path, dur=5.0, sr=8000):
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.zeros(int(dur * sr), dtype=np.float32), sr)


@pytest.fixture
def server(tmp_path):
    root = tmp_path / "srv"
    repo = root / "own" / "repo" / "master"
    names = [f"{f}-{i}-A-0.wav" for f in (1, 2) for i in range(3)]
    (repo / "meta").mkdir(parents=True)
    pd.DataFrame({"filename": names, "fold": [1, 1, 1, 2, 2, 2], "target": 0, "category": "dog",
                  "esc10": False, "src_file": range(6), "take": "A"}).to_csv(repo / "meta" / "esc50.csv", index=False)
    (repo / "LICENSE").write_text("lic")
    (repo / "README.md").write_text("readme")
    for n in names:
        _wav(repo / "audio" / n)
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(root))
    handler.log_message = lambda *a, **k: None  # type: ignore[attr-defined]
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}", repo, names
    srv.shutdown()


def test_fetch_all_and_resume(server, tmp_path):
    base, repo, names = server
    dest = tmp_path / "esc50"
    assert fetch_esc50("own/repo", dest, base=base, backoff_s=0) == 6
    assert (dest / "meta" / "esc50.csv").exists() and (dest / "LICENSE").exists()
    # corrupt one file and delete another: re-run repairs exactly those
    (dest / "audio" / names[0]).write_bytes(b"junk")
    (dest / "audio" / names[1]).unlink()
    assert fetch_esc50("own/repo", dest, base=base, backoff_s=0) == 6
    assert abs(sf.info(str(dest / "audio" / names[0])).duration - 5.0) < 0.01


def test_fetch_folds_subset(server, tmp_path):
    base, _, _ = server
    dest = tmp_path / "esc50"
    assert fetch_esc50("own/repo", dest, folds=(2,), base=base, backoff_s=0) == 3
    assert count_audio(dest) == 3


def test_fetch_reports_failures(server, tmp_path):
    base, repo, names = server
    (repo / "audio" / names[2]).unlink()  # listed in meta but 404 on the server
    with pytest.raises(RuntimeError, match="1 of 6 files failed"):
        fetch_esc50("own/repo", tmp_path / "esc50", base=base, retries=2, backoff_s=0)


def test_invalid_download_is_rejected(server, tmp_path):
    base, repo, names = server
    _wav(repo / "audio" / names[3], dur=2.0)  # wrong length on the server
    with pytest.raises(RuntimeError, match="1 of 6 files failed"):
        fetch_esc50("own/repo", tmp_path / "esc50", base=base, retries=2, backoff_s=0)
    assert not (tmp_path / "esc50" / "audio" / names[3]).exists()
