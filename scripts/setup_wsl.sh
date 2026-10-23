#!/usr/bin/env bash
# Run inside WSL2 Ubuntu from ~/ml_mini/p1_drone_sentinel
set -euo pipefail

sudo apt update
sudo apt install -y python3-pip git libsndfile1 portaudio19-dev ffmpeg software-properties-common

if ! command -v python3.11 >/dev/null; then
  # As far as I know Ubuntu 22.04 and 24.04 do not ship 3.11 in the default repos.
  sudo add-apt-repository -y ppa:deadsnakes/ppa
  sudo apt update
fi
sudo apt install -y python3.11 python3.11-venv

python3.11 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install -e .

echo "--- audio devices (if none show up, use FileSource or TcpSource, do not debug) ---"
.venv/bin/python -c "import sounddevice as sd; print(sd.query_devices())" || true
