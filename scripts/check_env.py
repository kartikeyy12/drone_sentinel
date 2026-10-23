#!/usr/bin/env python
"""M0 environment check for WSL2: python, packages, system libs, audio devices, disk, network."""
from __future__ import annotations

import importlib
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

PKGS = ["numpy", "scipy", "sklearn", "librosa", "soundfile", "sounddevice", "matplotlib", "pandas",
        "yaml", "zmq", "jinja2", "tqdm", "huggingface_hub", "pyarrow", "rich", "pytest", "joblib"]
URLS = {"huggingface.co": "https://huggingface.co", "github.com": "https://github.com",
        "figshare": "https://api.figshare.com/v2/articles?page_size=1"}


def line(ok: bool, name: str, detail: str = "") -> bool:
    print(f"[{'OK' if ok else 'FAIL'}] {name} {detail}")
    return ok


def main() -> int:
    bad = 0
    v = sys.version_info
    bad += not line(v[:2] == (3, 11), "python", f"{v.major}.{v.minor}.{v.micro} (spec: 3.11)")
    in_venv = sys.prefix != sys.base_prefix
    bad += not line(in_venv, "virtualenv active", sys.prefix)
    wsl = "microsoft" in Path("/proc/version").read_text().lower() if Path("/proc/version").exists() else False
    line(wsl, "running in WSL", "(informational)")
    root = Path(__file__).resolve().parents[1]
    bad += not line(not str(root).startswith("/mnt/"), "project on Linux filesystem", str(root))
    for p in PKGS:
        try:
            m = importlib.import_module(p)
            line(True, p, getattr(m, "__version__", ""))
        except Exception as e:  # noqa: BLE001
            bad += not line(False, p, str(e).splitlines()[0])
    for exe in ["ffmpeg", "git"]:
        bad += not line(shutil.which(exe) is not None, exe)
    ld = subprocess.run(["ldconfig", "-p"], capture_output=True, text=True).stdout
    for lib in ["libsndfile", "libportaudio"]:
        bad += not line(lib in ld, lib)
    try:
        import sounddevice as sd

        devs = [d for d in sd.query_devices() if d["max_input_channels"] > 0]
        line(bool(devs), "input audio devices", f"{len(devs)} found (if 0, use FileSource/TcpSource)")
    except Exception as e:  # noqa: BLE001
        line(False, "sounddevice query", str(e).splitlines()[0])
    free_gb = shutil.disk_usage(root).free / 1e9
    line(free_gb > 5, "free disk", f"{free_gb:.1f} GB")
    for name, url in URLS.items():
        try:
            urllib.request.urlopen(url, timeout=15)
            line(True, f"reach {name}")
        except Exception as e:  # noqa: BLE001
            line(False, f"reach {name}", str(e)[:80])
    print(f"\n{bad} blocking problem(s)")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
