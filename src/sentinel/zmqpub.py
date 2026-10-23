"""ZeroMQ PUB stream of closed events for another system (for example a multi-sensor counter-UAS fusion node).

Each closed event is sent as one two-frame message: frame 0 is the topic (default b"acoustic"), frame 1 is the event as UTF-8
JSON with the fields of the event schema. A subscriber:

    s = zmq.Context().socket(zmq.SUB); s.connect("tcp://127.0.0.1:5556"); s.setsockopt(zmq.SUBSCRIBE, b"acoustic")
    topic, payload = s.recv_multipart(); event = json.loads(payload)

PUB/SUB drops messages for subscribers that are not connected yet (and for ones that connect later): a subscriber sees only the
events that close after it connected. The events.jsonl file of the run has all of them.
"""
from __future__ import annotations

import json
import logging
from typing import Any

log = logging.getLogger("zmqpub")


class EventPublisher:
    """Binds a PUB socket on `endpoint` (use tcp://127.0.0.1:* to let the OS pick a port; see `.endpoint`)."""

    def __init__(self, endpoint: str = "tcp://127.0.0.1:5556", topic: str = "acoustic", context: Any = None) -> None:
        import zmq                                     # imported here so the rest of the package works without pyzmq

        self._zmq = zmq
        self._own_ctx = context is None
        self.ctx = context or zmq.Context()
        self.sock = self.ctx.socket(zmq.PUB)
        self.sock.setsockopt(zmq.LINGER, 1000)         # wait up to 1 s for queued events when closing
        self.sock.setsockopt(zmq.SNDHWM, 1000)
        self.sock.bind(endpoint)
        self.endpoint = self.sock.getsockopt(zmq.LAST_ENDPOINT).decode()
        self.topic = topic.encode("utf-8")
        self.n_sent = 0
        self.n_failed = 0
        self._closed = False

    def publish(self, rec: dict[str, Any]) -> bool:
        """Send one event record. Never raises: a failed send is counted and logged so the monitor keeps running."""
        if self._closed:
            return False
        try:
            self.sock.send_multipart([self.topic, json.dumps(rec).encode("utf-8")], flags=self._zmq.NOBLOCK)
            self.n_sent += 1
            return True
        except Exception as e:                         # noqa: BLE001 - a publisher problem must not stop the sentinel
            self.n_failed += 1
            log.warning("zmq publish failed: %s", e)
            return False

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.sock.close()
        if self._own_ctx:
            self.ctx.term()

    def __enter__(self) -> "EventPublisher":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
