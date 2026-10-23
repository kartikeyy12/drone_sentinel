"""Audio sources: file replay and pacing, TCP wire format and partial reads, fake PortAudio, the Windows streaming tool."""
from __future__ import annotations

import importlib.util
import socket
import sys
import threading
import time
import wave
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from sentinel import sources as src
from sentinel.config import load_config

SR = 16000
CFG = load_config()
TOOL = Path(__file__).resolve().parents[1] / "tools" / "win_mic_stream.py"


def collect(source, limit_s=10.0):
    out, t0 = [], time.monotonic()
    for c in source.chunks():
        out.append(c)
        if time.monotonic() - t0 > limit_s:
            break
    return np.concatenate(out) if out else np.zeros(0, np.float32)


def pcm(x):
    return np.round(np.clip(x, -1, 1) * 32767).astype("<i2")


# ---- file ------------------------------------------------------------------------------------------------------------------

def test_file_source_yields_the_whole_file_in_chunks_and_resamples(tmp_path):
    x = (0.3 * np.sin(2 * np.pi * 440 * np.arange(44100 * 2) / 44100)).astype(np.float32)
    sf.write(tmp_path / "a.wav", x, 44100)
    s = src.FileSource(tmp_path / "a.wav", SR, chunk_s=0.25)
    chunks = list(s.chunks())
    assert all(len(c) == 4000 for c in chunks[:-1]) and len(chunks) == 8
    assert sum(len(c) for c in chunks) == 2 * SR and s.duration_s == pytest.approx(2.0)
    assert s.describe() == "file:a.wav"


def test_file_source_missing_file_is_a_clear_error(tmp_path):
    with pytest.raises(src.SourceUnavailable, match="not found"):
        src.FileSource(tmp_path / "nope.wav", SR)


def test_file_source_realtime_waits_for_each_chunk_to_exist():
    t = [0.0]
    slept = []

    def sleep(d):
        slept.append(d)
        t[0] += d
    s = src.FileSource("mem.wav", SR, chunk_s=0.5, realtime=True, clock=lambda: t[0], sleep=sleep, samples=np.zeros(SR * 3, np.float32))
    n = sum(1 for _ in s.chunks())
    assert n == 6 and sum(slept) == pytest.approx(3.0) and slept[0] == pytest.approx(0.5)
    fast = src.FileSource("mem.wav", SR, chunk_s=0.5, realtime=False, clock=lambda: t[0], sleep=sleep, samples=np.zeros(SR, np.float32))
    slept.clear()
    list(fast.chunks())
    assert slept == []


# ---- tcp -------------------------------------------------------------------------------------------------------------------

def _send(port, payload, piece, delay=0.0):
    def run():
        c = socket.create_connection(("127.0.0.1", port), timeout=5)
        for i in range(0, len(payload), piece):
            c.sendall(payload[i:i + piece])
            if delay:
                time.sleep(delay)
        c.close()
    t = threading.Thread(target=run, daemon=True)
    t.start()
    return t


def test_tcp_source_reassembles_int16_even_when_samples_are_split_across_packets():
    x = (0.5 * np.sin(2 * np.pi * 300 * np.arange(SR * 2) / SR)).astype(np.float32)
    raw = pcm(x)
    s = src.TcpSource(SR, chunk_s=0.25, port=0)
    t = _send(s.port, raw.tobytes(), piece=3001)                      # odd size: a sample straddles two sends
    got = collect(s)
    t.join()
    s.close()
    assert len(got) == len(x)
    np.testing.assert_allclose(got, raw.astype(np.float32) / 32768.0, atol=0)
    assert s.dropped_samples == 0 and s.client is not None


def test_tcp_source_ends_when_the_client_disconnects_without_sending():
    s = src.TcpSource(SR, chunk_s=0.25, port=0)
    t = _send(s.port, b"", piece=1)
    assert collect(s, limit_s=5).size == 0
    t.join()
    s.close()


def test_tcp_source_port_in_use_is_a_clear_error():
    a = src.TcpSource(SR, port=0)
    try:
        with pytest.raises(src.SourceUnavailable, match="cannot listen"):
            src.TcpSource(SR, port=a.port)
    finally:
        a.close()


def test_tcp_source_counts_dropped_audio_when_the_consumer_stalls():
    s = src.TcpSource(SR, chunk_s=0.25, port=0, queue_s=0.5)         # room for 2 chunks
    raw = pcm(np.zeros(SR * 4, np.float32)).tobytes()
    t = _send(s.port, raw, piece=8000)
    gen = s.chunks()
    first = next(gen)                                                 # starts the reader, then stop consuming
    t.join()
    time.sleep(0.5)
    s.close()
    assert len(first) == 4000 and s.dropped_samples > 0


def test_windows_tool_streams_a_wav_in_the_wire_format(tmp_path):
    spec = importlib.util.spec_from_file_location("win_mic_stream", TOOL)
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    x = (0.4 * np.sin(2 * np.pi * 500 * np.arange(SR * 3) / SR)).astype(np.float32)
    path = tmp_path / "t.wav"
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1), w.setsampwidth(2), w.setframerate(SR)
        w.writeframes(pcm(x).tobytes())
    s = src.TcpSource(SR, chunk_s=0.25, port=0)

    def client():
        sock = socket.create_connection(("127.0.0.1", s.port), timeout=5)
        tool.stream_wav(sock, str(path), realtime=False)
        sock.close()
    t = threading.Thread(target=client, daemon=True)
    t.start()
    got = collect(s)
    t.join()
    s.close()
    np.testing.assert_allclose(got, pcm(x).astype(np.float32) / 32768.0, atol=0)


def test_windows_tool_rejects_a_wav_with_the_wrong_format(tmp_path):
    spec = importlib.util.spec_from_file_location("win_mic_stream", TOOL)
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    path = tmp_path / "bad.wav"
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1), w.setsampwidth(2), w.setframerate(44100)
        w.writeframes(b"\0\0" * 100)
    with pytest.raises(SystemExit, match="16 kHz mono"):
        tool.stream_wav(None, str(path), False)


# ---- sounddevice -----------------------------------------------------------------------------------------------------------

class FakeSD:
    """Stands in for the sounddevice module: a device list and an InputStream that calls the callback from a thread."""

    def __init__(self, devices, blocks=8, block=4000, burst=False):
        self._devices, self.blocks, self.block, self.burst = devices, blocks, block, burst
        self.opened = {}
        self.stream = None

    def query_devices(self):
        return self._devices

    def InputStream(self, **kw):
        self.opened = kw
        outer = self

        class S:
            def __init__(self):
                self.stopped = self.closed = False

            def start(self):
                def feed():
                    for i in range(outer.blocks):
                        data = np.full((outer.block, 1), (i + 1) / 100.0, np.float32)
                        kw["callback"](data, outer.block, None, "overflow" if i == 0 else None)
                        if not outer.burst:
                            time.sleep(0.005)
                threading.Thread(target=feed, daemon=True).start()

            def stop(self):
                self.stopped = True

            def close(self):
                self.closed = True
        outer.stream = S()
        return outer.stream


def test_sounddevice_source_streams_callback_blocks_in_order():
    fake = FakeSD([{"max_input_channels": 1}], blocks=6)
    s = src.SoundDeviceSource(SR, chunk_s=0.25, sd_module=fake, device=3)
    got = []
    for c in s.chunks():
        got.append(c)
        if len(got) == 6:
            break
    s.close()
    assert [round(float(c[0]) * 100) for c in got] == [1, 2, 3, 4, 5, 6]
    assert fake.opened["samplerate"] == SR and fake.opened["channels"] == 1 and fake.opened["device"] == 3
    assert fake.stream.stopped and fake.stream.closed and s.n_status_events == 1 and s.name == "mic:3"


def test_sounddevice_source_counts_dropped_audio_when_the_queue_is_full():
    fake = FakeSD([{"max_input_channels": 2}], blocks=10, burst=True)
    s = src.SoundDeviceSource(SR, chunk_s=0.25, queue_s=0.5, sd_module=fake)          # room for 2 chunks
    gen = s.chunks()
    next(gen)
    time.sleep(0.2)
    s.close()
    assert 7 * 4000 <= s.dropped_samples <= 8 * 4000        # 10 blocks arrived at once, 2 fit, 1 may have been taken meanwhile


def test_sounddevice_source_without_an_input_device_says_what_to_do():
    with pytest.raises(src.SourceUnavailable, match="--source tcp"):
        src.SoundDeviceSource(SR, sd_module=FakeSD([{"max_input_channels": 0}]))
    with pytest.raises(src.SourceUnavailable, match="--source tcp"):
        src.SoundDeviceSource(SR, sd_module=FakeSD([]))


def test_sounddevice_source_when_portaudio_cannot_load(monkeypatch):
    monkeypatch.setitem(sys.modules, "sounddevice", None)             # `import sounddevice` now raises ImportError
    with pytest.raises(src.SourceUnavailable, match="PortAudio"):
        src.SoundDeviceSource(SR)


def test_sounddevice_source_open_failure_is_reported():
    class Broken(FakeSD):
        def InputStream(self, **kw):
            raise OSError("Invalid sample rate")
    s = src.SoundDeviceSource(SR, sd_module=Broken([{"max_input_channels": 1}]))
    with pytest.raises(src.SourceUnavailable, match="Invalid sample rate"):
        next(s.chunks())


# ---- factory ---------------------------------------------------------------------------------------------------------------

def test_open_source_dispatch(tmp_path):
    sf.write(tmp_path / "a.wav", np.zeros(SR, np.float32), SR)
    f = src.open_source("file", CFG, file=str(tmp_path / "a.wav"), realtime=True)
    assert isinstance(f, src.FileSource) and f.realtime
    with pytest.raises(src.SourceUnavailable, match="--file"):
        src.open_source("file", CFG)
    tcp = src.open_source("tcp", CFG, host="127.0.0.1", port=_free_port())
    try:
        assert isinstance(tcp, src.TcpSource)
    finally:
        tcp.close()
    with pytest.raises(src.SourceUnavailable, match="unknown source"):
        src.open_source("radio", CFG)


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p
