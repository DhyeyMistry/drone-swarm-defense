"""telemetry: live data any policy publishes and any other policy reads, in or out of process.

Topics are strings ("<policy>/<what>", e.g. "formation/drone4"); a value is any JSON-able
object. Only the latest value per topic is kept, with its publish time: a reader wants the
freshest data and its age, not a backlog.

  bus.publish("spin/drone1", {"yaw_rc": 40})
  hit = bus.latest("formation/drone4")        # (data, age_s) or None
  bus.since("formation/")                     # {topic: (data, age_s)}

In swarm_policies.py every policy gets the umbrella's bus as self.bus. A policy in its own
process (e.g. swarm_formation.py) uses TelemetryClient.from_env(): same calls, carried over
localhost UDP to the umbrella's TelemetryHub, which relays every publish to every other
process. Times are time.time() on this machine. Stdlib only.
"""
import json
import os
import socket
import threading
import time

ENV = "SWARM_TELEM"          # "host:port" of the umbrella's hub, set for external policies
MAX_DGRAM = 65000


class TelemetryBus:
    """Latest value per topic, thread-safe. Listeners get every publish (topic, data, t, src)."""

    def __init__(self):
        self._lock = threading.Lock()
        self._data = {}
        self.listeners = []

    def publish(self, topic, data, t=None, src=None):
        t = time.time() if t is None else t
        with self._lock:
            old = self._data.get(topic)
            if old is not None and old[1] > t:
                return                               # out-of-order datagram: keep the newer
            self._data[topic] = (data, t)
        for fn in self.listeners:
            fn(topic, data, t, src)

    def latest(self, topic, max_age=None):
        """(data, age_s), or None if never published or older than max_age."""
        with self._lock:
            hit = self._data.get(topic)
        if hit is None:
            return None
        age = time.time() - hit[1]
        return None if max_age is not None and age > max_age else (hit[0], age)

    def since(self, prefix="", max_age=None):
        """{topic: (data, age_s)} for every topic starting with prefix."""
        now = time.time()
        with self._lock:
            items = [(k, v) for k, v in self._data.items() if k.startswith(prefix)]
        return {k: (d, now - t) for k, (d, t) in items if max_age is None or now - t <= max_age}


def _encode(topic, data, t):
    msg = json.dumps({"topic": topic, "t": t, "data": data}, separators=(",", ":")).encode()
    if len(msg) > MAX_DGRAM:
        raise ValueError(f"telemetry {topic!r}: {len(msg)} bytes > {MAX_DGRAM}")
    return msg


class TelemetryHub:
    """The umbrella's end: a localhost UDP socket in front of its bus. A process says hello
    once, then publishes; the hub relays every publish (in-process ones too) to every other
    process."""

    def __init__(self, bus, host="127.0.0.1", port=0):
        self.bus = bus
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((host, port))
        self.addr = self.sock.getsockname()
        self.clients = set()
        self._lock = threading.Lock()
        bus.listeners.append(self._relay)
        threading.Thread(target=self._loop, daemon=True).start()

    def env(self):
        """Environment for an external policy's process."""
        return {**os.environ, ENV: f"{self.addr[0]}:{self.addr[1]}"}

    def _loop(self):
        while True:
            try:
                raw, src = self.sock.recvfrom(MAX_DGRAM)
                msg = json.loads(raw)
            except OSError:
                return                               # closed
            except ValueError:
                continue
            with self._lock:
                new = src not in self.clients
                self.clients.add(src)
            if new:                                  # catch it up on everything so far
                for topic, (data, age) in self.bus.since().items():
                    self._send(_encode(topic, data, time.time() - age), src)
            if msg.get("topic"):
                self.bus.publish(msg["topic"], msg.get("data"), msg.get("t"), src)

    def _relay(self, topic, data, t, src):
        msg = _encode(topic, data, t)
        with self._lock:
            dests = [c for c in self.clients if c != src]
        for c in dests:
            self._send(msg, c)

    def _send(self, msg, dest):
        try:
            self.sock.sendto(msg, dest)
        except OSError:
            pass

    def close(self):
        self.sock.close()


class TelemetryClient(TelemetryBus):
    """An external policy's end: a local bus kept in sync with the umbrella's hub."""

    def __init__(self, hub_addr):
        super().__init__()
        self.hub = hub_addr
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("127.0.0.1", 0))
        self.sock.sendto(b"{}", self.hub)            # hello: start relaying to me
        threading.Thread(target=self._loop, daemon=True).start()

    @classmethod
    def from_env(cls):
        """A client if launched by the umbrella, else a plain local bus (publishing is a no-op
        for anyone else)."""
        spec = os.environ.get(ENV)
        if not spec:
            return TelemetryBus()
        host, _, port = spec.rpartition(":")
        return cls((host, int(port)))

    def publish(self, topic, data, t=None, src=None):
        t = time.time() if t is None else t
        super().publish(topic, data, t, src)
        if src is None:                              # ours, not relayed from the hub
            try:
                self.sock.sendto(_encode(topic, data, t), self.hub)
            except OSError:
                pass

    def _loop(self):
        while True:
            try:
                msg = json.loads(self.sock.recv(MAX_DGRAM))
            except OSError:
                return
            except ValueError:
                continue
            if msg.get("topic"):
                super().publish(msg["topic"], msg.get("data"), msg.get("t"), src="hub")

    def close(self):
        self.sock.close()
