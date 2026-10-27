"""Receive the Tello's raw H.264 UDP stream and decode it with PyAV.

Only the most recent decoded frame is kept, so a slow consumer never builds latency.
"""

from __future__ import annotations

import logging
import socket
import threading
import time

import av
import av.logging
import numpy as np

log = logging.getLogger(__name__)

VIDEO_PORT = 11111

# libav prints "non-existing PPS" etc. until the first keyframe; that is expected.
av.logging.set_level(av.logging.PANIC)


class VideoReceiver:
    def __init__(self, local_ip: str, port: int = VIDEO_PORT) -> None:
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)
        self._sock.bind((local_ip, port))
        self._sock.settimeout(0.5)
        self._codec = av.CodecContext.create("h264", "r")
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._frame_id = 0
        self._t_recv: float | None = None
        self.bytes_received = 0
        self.decode_errors = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="tello-video", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def get_latest(self) -> tuple[np.ndarray | None, int, float | None]:
        """Return (BGR frame, frame_id, t_recv).

        frame_id increases by one per decoded frame; t_recv is time.monotonic() when
        the UDP datagram that completed the frame arrived.
        """
        with self._lock:
            return self._frame, self._frame_id, self._t_recv

    def stop(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._sock.close()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                data, _ = self._sock.recvfrom(65536)
            except socket.timeout:
                continue
            except OSError:
                break  # socket closed
            t_recv = time.monotonic()
            self.bytes_received += len(data)
            self._decode(data, t_recv)

    def _decode(self, data: bytes, t_recv: float) -> None:
        try:
            for packet in self._codec.parse(data):
                for frame in self._codec.decode(packet):
                    image = frame.to_ndarray(format="bgr24")
                    with self._lock:
                        self._frame = image
                        self._frame_id += 1
                        self._t_recv = t_recv
        except av.FFmpegError as exc:
            self.decode_errors += 1
            log.debug("decode error (normal before first keyframe): %s", exc)
