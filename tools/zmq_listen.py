#!/usr/bin/env python
"""Print the events of a running `python -m sentinel.monitor ... --zmq`. A stand-in for the fusion node that would consume them.

    python tools/zmq_listen.py [--endpoint tcp://127.0.0.1:5556] [--topic acoustic] [--count N] [--timeout S]

Start this BEFORE the monitor: PUB/SUB does not replay events that closed before the subscriber connected.
Exit code 0 when at least one event was received (or --count reached), 1 on timeout with none.
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import zmq


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--endpoint", default="tcp://127.0.0.1:5556")
    ap.add_argument("--topic", default="acoustic")
    ap.add_argument("--count", type=int, default=0, help="stop after this many events (0 = until Ctrl+C or timeout)")
    ap.add_argument("--timeout", type=float, default=0.0, help="stop after this many seconds without an event (0 = wait forever)")
    a = ap.parse_args(argv)
    s = zmq.Context.instance().socket(zmq.SUB)
    s.setsockopt(zmq.SUBSCRIBE, a.topic.encode())
    s.connect(a.endpoint)
    print(f"listening on {a.endpoint}, topic {a.topic}", flush=True)
    n, last = 0, time.monotonic()
    try:
        while True:
            if s.poll(200):
                topic, payload = s.recv_multipart()
                print(topic.decode(), json.dumps(json.loads(payload)), flush=True)
                n, last = n + 1, time.monotonic()
                if a.count and n >= a.count:
                    break
            elif a.timeout and time.monotonic() - last > a.timeout:
                break
    except KeyboardInterrupt:
        pass
    finally:
        s.close()
    return 0 if n else 1


if __name__ == "__main__":
    sys.exit(main())
