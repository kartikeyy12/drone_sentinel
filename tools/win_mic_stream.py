#!/usr/bin/env python
"""Stream a microphone to the sentinel monitor over TCP. Run this with WINDOWS Python, not inside WSL.

    py -m pip install sounddevice
    py win_mic_stream.py --list-devices
    py win_mic_stream.py --device 1 --host 127.0.0.1 --port 5005

The monitor (in WSL) must already be waiting:  python -m sentinel.monitor --source tcp

Wire format: raw little-endian int16, mono, 16000 Hz, no header. Ctrl+C stops it; the monitor then flushes its open event.
`--wav file.wav` sends a 16 kHz mono 16-bit PCM WAV instead of the microphone (standard library only), which is a quick
way to test the connection without a microphone, from Windows or from WSL.
If localhost does not reach WSL2, run `hostname -I` in WSL and pass that address with --host and the monitor's --tcp-host 0.0.0.0.
"""
from __future__ import annotations

import argparse
import queue
import socket
import sys
import time
import wave

SR = 16000


def stream_wav(sock: socket.socket, path: str, realtime: bool) -> None:
    with wave.open(path, "rb") as w:
        if (w.getframerate(), w.getnchannels(), w.getsampwidth()) != (SR, 1, 2):
            sys.exit(f"{path}: need 16 kHz mono 16-bit PCM, got {w.getframerate()} Hz, {w.getnchannels()} ch, {8 * w.getsampwidth()} bit")
        block = SR // 4
        t0 = time.monotonic()
        sent = 0
        while True:
            data = w.readframes(block)
            if not data:
                break
            sock.sendall(data)
            sent += len(data) // 2
            if realtime:
                time.sleep(max(0.0, t0 + sent / SR - time.monotonic()))


def stream_mic(sock: socket.socket, device, seconds: float | None) -> None:
    try:
        import sounddevice as sd
    except ImportError:
        sys.exit("sounddevice is not installed: py -m pip install sounddevice")
    q: queue.Queue = queue.Queue()

    def cb(indata, frames, t, status):
        if status:
            print(status, file=sys.stderr)
        q.put(bytes(indata))

    t0 = time.monotonic()
    with sd.RawInputStream(samplerate=SR, channels=1, dtype="int16", device=device, blocksize=SR // 8, callback=cb):
        print("streaming, Ctrl+C to stop", flush=True)
        while seconds is None or time.monotonic() - t0 < seconds:
            try:
                sock.sendall(q.get(timeout=0.5))
            except queue.Empty:
                continue


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5005)
    ap.add_argument("--device", default=None, help="input device index or name substring (see --list-devices)")
    ap.add_argument("--list-devices", action="store_true")
    ap.add_argument("--seconds", type=float, default=None, help="stop after this long (default: until Ctrl+C)")
    ap.add_argument("--wav", default=None, help="send this WAV file instead of the microphone")
    ap.add_argument("--realtime", action="store_true", help="with --wav: pace the file like a live microphone")
    a = ap.parse_args()
    if a.list_devices:
        import sounddevice as sd
        print(sd.query_devices())
        return 0
    device = int(a.device) if a.device is not None and str(a.device).isdigit() else a.device
    try:
        sock = socket.create_connection((a.host, a.port), timeout=10)
    except OSError as e:
        print(f"cannot connect to {a.host}:{a.port} ({e}). Start the monitor first: python -m sentinel.monitor --source tcp")
        return 1
    sock.settimeout(None)
    try:
        if a.wav:
            stream_wav(sock, a.wav, a.realtime)
        else:
            stream_mic(sock, device, a.seconds)
    except KeyboardInterrupt:
        print("\nstopped")
    except (BrokenPipeError, ConnectionResetError):
        print("the monitor closed the connection")
    finally:
        sock.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
