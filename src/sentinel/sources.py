"""Audio sources for the live engine. Every source yields mono float32 chunks at the engine sample rate.

  FileSource         replays a WAV (any soundfile format). Default for development and tests; `realtime=True` paces it
                     like a microphone.
  SoundDeviceSource  a local input device through PortAudio (works in WSL2 only if WSLg exposes a PulseAudio input).
  TcpSource          listens on a TCP port; tools/win_mic_stream.py (run in Windows Python) streams the Windows microphone
                     to it as raw little-endian int16 mono at the engine sample rate.

Live sources buffer at most `queue_s` seconds. If the consumer falls behind, the oldest audio is kept, new chunks are
dropped and counted in `dropped_samples` (a dropped chunk is a gap in the stream, so event times after it are late by
that much; the monitor reports it).
"""
from __future__ import annotations

import logging
import queue
import socket
import threading
import time
from pathlib import Path
from typing import Callable, Iterator

import numpy as np

from .preprocess import load_audio

log = logging.getLogger(__name__)

INT16_SCALE = 32768.0


class SourceUnavailable(RuntimeError):
    """The source cannot be opened (no input device, port in use, ...). The message says what to do."""


def int16_to_float(raw: np.ndarray) -> np.ndarray:
    return (np.asarray(raw, np.int16).astype(np.float32) / INT16_SCALE)


class AudioSource:
    """Base class: `chunks()` yields float32 mono arrays; `close()` releases the device or socket."""

    name = "source"
    sr: int
    chunk_samples: int
    dropped_samples = 0

    def chunks(self) -> Iterator[np.ndarray]:  # pragma: no cover - interface
        raise NotImplementedError

    def close(self) -> None:
        pass

    def __enter__(self) -> "AudioSource":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def describe(self) -> str:
        return self.name


# ---- file ----------------------------------------------------------------------------------------------------------------

class FileSource(AudioSource):
    """Replay of an audio file. Without `realtime` the chunks come as fast as they are consumed."""

    def __init__(self, path: str | Path, sr: int, chunk_s: float = 0.25, realtime: bool = False,
                 clock: Callable[[], float] = time.monotonic, sleep: Callable[[float], None] = time.sleep,
                 samples: np.ndarray | None = None) -> None:
        self.path, self.sr, self.realtime = Path(path), int(sr), bool(realtime)
        self.chunk_samples = max(1, int(round(chunk_s * sr)))
        self._clock, self._sleep = clock, sleep
        if samples is not None:                       # in-memory audio, for tests and scenario runs
            self._x = np.asarray(samples, np.float32)
        else:
            if not self.path.exists():
                raise SourceUnavailable(f"audio file not found: {self.path}")
            self._x = load_audio(self.path, self.sr)
        self.name = f"file:{self.path.name}"
        self.duration_s = len(self._x) / self.sr

    def chunks(self) -> Iterator[np.ndarray]:
        t0 = self._clock()
        for k, i in enumerate(range(0, len(self._x), self.chunk_samples)):
            if self.realtime:
                due = t0 + i / self.sr + self.chunk_samples / self.sr     # a chunk exists once it has been recorded
                wait = due - self._clock()
                if wait > 0:
                    self._sleep(wait)
            yield self._x[i:i + self.chunk_samples].copy()


# ---- local microphone ------------------------------------------------------------------------------------------------------

class SoundDeviceSource(AudioSource):
    """Input device through `sounddevice`. The PortAudio callback only copies into a queue."""

    def __init__(self, sr: int, chunk_s: float = 0.25, queue_s: float = 30.0, device: int | str | None = None,
                 sd_module=None) -> None:
        self.sr, self.device = int(sr), device
        self.chunk_samples = max(1, int(round(chunk_s * sr)))
        self._max_chunks = max(1, int(round(queue_s / chunk_s)))
        self._q: queue.Queue[np.ndarray] = queue.Queue(maxsize=self._max_chunks)
        self._stop = threading.Event()
        self._stream = None
        self.n_status_events = 0
        try:
            if sd_module is None:
                import sounddevice as sd_module  # PortAudio is loaded at import time
        except (ImportError, OSError) as e:
            raise SourceUnavailable(f"sounddevice / PortAudio is not usable here ({e}). In WSL2 try: sudo apt install libportaudio2 "
                                    "pulseaudio-utils; or stream the Windows microphone with --source tcp") from e
        self._sd = sd_module
        try:
            devices = self._sd.query_devices()
            if not any(d.get("max_input_channels", 0) > 0 for d in devices):
                raise SourceUnavailable("no audio input device found (WSLg gives WSL2 a PulseAudio input only on some "
                                        "setups). Use --source tcp with tools/win_mic_stream.py on Windows instead")
        except SourceUnavailable:
            raise
        except Exception as e:  # PortAudioError and friends
            raise SourceUnavailable(f"could not query audio devices: {e}") from e
        self.name = f"mic:{device if device is not None else 'default'}"

    def _callback(self, indata, frames, time_info, status) -> None:   # runs in the PortAudio thread
        if status:
            self.n_status_events += 1
        try:
            self._q.put_nowait(np.asarray(indata[:, 0], np.float32).copy())
        except queue.Full:
            self.dropped_samples += frames

    def chunks(self) -> Iterator[np.ndarray]:
        try:
            self._stream = self._sd.InputStream(samplerate=self.sr, channels=1, dtype="float32", device=self.device,
                                                blocksize=self.chunk_samples, callback=self._callback)
            self._stream.start()
        except Exception as e:
            raise SourceUnavailable(f"could not open the input device ({e})") from e
        while not self._stop.is_set():
            try:
                yield self._q.get(timeout=0.2)
            except queue.Empty:
                continue

    def close(self) -> None:
        self._stop.set()
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:  # pragma: no cover - device already gone
                pass
            self._stream = None


# ---- TCP -------------------------------------------------------------------------------------------------------------------

class TcpSource(AudioSource):
    """TCP server for one client that sends raw little-endian int16 mono audio at `sr`. End of stream when it disconnects."""

    def __init__(self, sr: int, chunk_s: float = 0.25, host: str = "127.0.0.1", port: int = 5005,
                 queue_s: float = 30.0, accept_timeout_s: float | None = None) -> None:
        self.sr, self.host = int(sr), host
        self.chunk_samples = max(1, int(round(chunk_s * sr)))
        self._max_chunks = max(1, int(round(queue_s / chunk_s)))
        self.accept_timeout_s = accept_timeout_s
        self._srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            self._srv.bind((host, int(port)))
        except OSError as e:
            self._srv.close()
            raise SourceUnavailable(f"cannot listen on {host}:{port} ({e}); is another monitor running?") from e
        self._srv.listen(1)
        self._srv.settimeout(0.2)
        self.port = self._srv.getsockname()[1]
        self.name = f"tcp:{host}:{self.port}"
        self._conn: socket.socket | None = None
        self._stop = threading.Event()
        self._q: queue.Queue[np.ndarray | None] = queue.Queue(maxsize=self._max_chunks)
        self._thread: threading.Thread | None = None
        self.client: str | None = None

    def _accept(self) -> socket.socket | None:
        t0 = time.monotonic()
        while not self._stop.is_set():
            try:
                conn, addr = self._srv.accept()
                self.client = f"{addr[0]}:{addr[1]}"
                return conn
            except socket.timeout:
                if self.accept_timeout_s is not None and time.monotonic() - t0 > self.accept_timeout_s:
                    return None
        return None

    def _reader(self) -> None:
        """Receive thread: bytes in, int16 chunks out, odd trailing bytes carried to the next recv."""
        try:
            conn = self._accept()
            if conn is None:
                return
            self._conn = conn
            conn.settimeout(0.2)
            log.info("tcp client connected from %s", self.client)
            carry, pending = b"", np.zeros(0, np.float32)
            while not self._stop.is_set():
                try:
                    data = conn.recv(65536)
                except socket.timeout:
                    continue
                except OSError:
                    break
                if not data:
                    break
                data = carry + data
                n = len(data) // 2 * 2
                carry = data[n:]
                if n:
                    pending = np.concatenate([pending, int16_to_float(np.frombuffer(data[:n], dtype="<i2"))])
                while len(pending) >= self.chunk_samples:
                    chunk, pending = pending[:self.chunk_samples], pending[self.chunk_samples:]
                    try:
                        self._q.put_nowait(chunk)
                    except queue.Full:
                        self.dropped_samples += len(chunk)
            if len(pending):
                try:
                    self._q.put_nowait(pending)
                except queue.Full:
                    self.dropped_samples += len(pending)
        finally:
            while not self._stop.is_set():
                try:
                    self._q.put(None, timeout=0.2)        # end-of-stream marker
                    break
                except queue.Full:
                    continue

    def chunks(self) -> Iterator[np.ndarray]:
        self._thread = threading.Thread(target=self._reader, name="tcp-reader", daemon=True)
        self._thread.start()
        while not self._stop.is_set():
            try:
                c = self._q.get(timeout=0.2)
            except queue.Empty:
                continue
            if c is None:
                return
            yield c

    def close(self) -> None:
        self._stop.set()
        for s in (self._conn, self._srv):
            try:
                if s is not None:
                    s.close()
            except OSError:  # pragma: no cover
                pass
        if self._thread is not None:
            self._thread.join(timeout=2.0)


def open_source(kind: str, cfg: dict, file: str | None = None, realtime: bool = False, device=None,
                host: str | None = None, port: int | None = None) -> AudioSource:
    """Build the source named on the command line (`mic`, `tcp` or `file`)."""
    sr, st, tcp = cfg["audio"]["sr"], cfg["m5"]["stream"], cfg["m5"]["tcp"]
    if kind == "file":
        if not file:
            raise SourceUnavailable("--source file needs --file path/to/audio.wav")
        return FileSource(file, sr, st["chunk_s"], realtime=realtime)
    if kind == "mic":
        return SoundDeviceSource(sr, st["chunk_s"], st["queue_s"], device=device)
    if kind == "tcp":
        return TcpSource(sr, st["chunk_s"], host or tcp["host"], int(port or tcp["port"]), st["queue_s"])
    raise SourceUnavailable(f"unknown source {kind!r}; choose mic, tcp or file")
