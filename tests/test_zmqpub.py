"""ZeroMQ event publisher: loopback subscriber, topic filtering, monitor wiring, failure isolation."""
from __future__ import annotations

import io
import json
import socket
import time

import numpy as np
import pytest
import soundfile as sf
from rich.console import Console

zmq = pytest.importorskip("zmq")

from helpers_stream import SR, scene  # noqa: E402
from sentinel import monitor as mon  # noqa: E402
from sentinel.bundle import save_model  # noqa: E402
from sentinel.sources import FileSource  # noqa: E402
from sentinel.zmqpub import EventPublisher  # noqa: E402

REC = {"sensor_id": "acoustic-01", "event_id": 7, "label": "aircraft", "t_start": "2026-10-12T10:15:03.000Z",
       "t_end": "2026-10-12T10:15:41.000Z", "duration_s": 38.0, "peak_prob": 0.97, "mean_prob": 0.91,
       "top_alt_class": "ambient", "snr_est_db": 9.4, "model_version": "abc"}


def subscriber(endpoint: str, topic: bytes = b"acoustic"):
    s = zmq.Context.instance().socket(zmq.SUB)
    s.setsockopt(zmq.SUBSCRIBE, topic)
    s.setsockopt(zmq.RCVTIMEO, 5000)
    s.connect(endpoint)
    time.sleep(0.4)                       # PUB/SUB slow joiner: let the subscription reach the publisher
    return s


def test_event_arrives_as_topic_and_json_frames():
    with EventPublisher("tcp://127.0.0.1:*") as pub:
        sub = subscriber(pub.endpoint)
        assert pub.endpoint.startswith("tcp://127.0.0.1:") and not pub.endpoint.endswith("*")
        assert pub.publish(REC)
        topic, payload = sub.recv_multipart()
        sub.close()
    assert topic == b"acoustic" and json.loads(payload) == REC and pub.n_sent == 1 and pub.n_failed == 0


def test_a_subscriber_on_another_topic_gets_nothing():
    with EventPublisher("tcp://127.0.0.1:*", topic="acoustic") as pub:
        sub = subscriber(pub.endpoint, b"radar")
        sub.setsockopt(zmq.RCVTIMEO, 300)
        pub.publish(REC)
        with pytest.raises(zmq.Again):
            sub.recv_multipart()
        sub.close()


def test_publish_after_close_is_a_harmless_false():
    pub = EventPublisher("tcp://127.0.0.1:*")
    pub.close()
    pub.close()
    assert pub.publish(REC) is False


def test_monitor_run_publishes_exactly_the_events_it_writes(toy, tmp_path):
    cfg, model = toy
    x = scene(8, 10, 8)
    with EventPublisher("tcp://127.0.0.1:*") as pub:
        sub = subscriber(pub.endpoint)
        sf.write(tmp_path / "s.wav", x, SR)
        src = FileSource(tmp_path / "s.wav", SR, chunk_s=0.25)
        s = mon.run(src, model, cfg, runs_dir=tmp_path / "runs", ui=False, t0_wall=1_760_000_000.0, console=Console(file=io.StringIO()),
                    publisher=pub)
        got = [json.loads(sub.recv_multipart()[1]) for _ in range(s["n_events"])]
        sub.close()
    written = [json.loads(l) for l in (tmp_path / "runs" / s["run_dir"].split("/")[-1] / "events.jsonl").read_text().splitlines()]
    assert s["n_events"] == 1 and got == written
    assert s["zmq"]["sent"] == 1 and s["zmq"]["failed"] == 0


def test_a_broken_publisher_does_not_stop_the_monitor(toy, tmp_path):
    cfg, model = toy

    class Broken:
        endpoint, n_sent, n_failed = "x", 0, 0

        def publish(self, rec):
            self.n_failed += 1
            return False

    sf.write(tmp_path / "s.wav", scene(8, 10, 8), SR)
    s = mon.run(FileSource(tmp_path / "s.wav", SR, chunk_s=0.25), model, cfg, runs_dir=tmp_path / "runs", ui=False,
                console=Console(file=io.StringIO()), publisher=Broken())
    assert s["status"] == "finished" and s["n_events"] == 1 and s["zmq"]["failed"] == 1


def test_main_zmq_flag_starts_the_publisher_and_reports_it(toy, tmp_path, capsys):
    cfg, model = toy
    p = tmp_path / "models" / "pipeline.joblib"
    save_model(model, p)
    sf.write(tmp_path / "scene.wav", scene(8, 10, 8), SR)
    with socket.socket() as sk:
        sk.bind(("127.0.0.1", 0))
        port = sk.getsockname()[1]
    rc = mon.main(["--source", "file", "--file", str(tmp_path / "scene.wav"), "--model", str(p), "--runs-dir", str(tmp_path / "runs"),
                   "--no-ui", "--zmq", "--zmq-endpoint", f"tcp://127.0.0.1:{port}"])
    out = capsys.readouterr().out
    assert rc == 0 and f"publishing events on tcp://127.0.0.1:{port}" in out and "ZeroMQ: 1 events published, 0 failed" in out


def test_main_zmq_port_in_use_is_reported_not_a_traceback(toy, tmp_path, capsys):
    cfg, model = toy
    p = tmp_path / "models" / "pipeline.joblib"
    save_model(model, p)
    sf.write(tmp_path / "scene.wav", scene(8, 10, 8), SR)
    with EventPublisher("tcp://127.0.0.1:*") as busy:
        rc = mon.main(["--source", "file", "--file", str(tmp_path / "scene.wav"), "--model", str(p), "--runs-dir", str(tmp_path / "runs"),
                       "--no-ui", "--zmq", "--zmq-endpoint", busy.endpoint])
    assert rc == 1 and "cannot start the ZeroMQ publisher" in capsys.readouterr().out


def test_listener_tool_prints_the_events_it_receives(tmp_path):
    import subprocess
    import sys
    from pathlib import Path
    tool = Path(__file__).resolve().parents[1] / "tools" / "zmq_listen.py"
    with EventPublisher("tcp://127.0.0.1:*") as pub:
        p = subprocess.Popen([sys.executable, str(tool), "--endpoint", pub.endpoint, "--count", "2", "--timeout", "10"],
                             stdout=subprocess.PIPE, text=True)
        assert p.stdout.readline().startswith("listening on")
        time.sleep(0.6)                                   # subscription reaches the publisher
        pub.publish(REC)
        pub.publish({**REC, "event_id": 8})
        out, _ = p.communicate(timeout=20)
    lines = out.strip().splitlines()
    assert p.returncode == 0 and [json.loads(l.split(" ", 1)[1])["event_id"] for l in lines] == [7, 8] and lines[0].startswith("acoustic ")


def test_listener_tool_times_out_with_exit_code_1():
    import subprocess
    import sys
    from pathlib import Path
    tool = Path(__file__).resolve().parents[1] / "tools" / "zmq_listen.py"
    with EventPublisher("tcp://127.0.0.1:*") as pub:
        r = subprocess.run([sys.executable, str(tool), "--endpoint", pub.endpoint, "--timeout", "1"], capture_output=True, text=True, timeout=20)
    assert r.returncode == 1
