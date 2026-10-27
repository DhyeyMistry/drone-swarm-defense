"""Attacker feeds: drone_gui telemetry and the swarm_policies.py telemetry bus."""

import json
import socket
import time

import numpy as np

from feeds import SwarmBusFeed, TelemetryFeed, display_name, drone_number, entry_to_map


def test_drone_names_in_any_form():
    table = {"TELLO-8": 8}
    assert drone_number("TELLO-8", table) == 8
    assert drone_number("drone8", table) == 8
    assert drone_number("8", table) == 8
    assert display_name(8, table) == "TELLO-8"


def test_drone_gui_pose_converts_through_the_map():
    # tag 6 on the far wall, 7 m out, facing back into the room
    M6 = np.eye(4)
    M6[:3, :3] = [[-1, 0, 0], [0, 1, 0], [0, 0, -1]]
    M6[:3, 3] = [0.5, 0.0, 7.0]
    poses = {10: np.eye(4), 6: M6}
    # README convention: x right, y down, z into the wall. 2 m in front of tag 6,
    # 1 m to its right (as the attacker sees it), 0.3 m below it.
    p = entry_to_map({"pose": {"tag": 6, "xyz": [1.0, 0.3, -2.0]}}, poses)
    assert np.allclose(p, [-0.5, -0.3, 5.0])
    # a ready map-frame position passes straight through
    assert np.allclose(entry_to_map({"map_xyz": [1, 2, 3]}, poses), [1, 2, 3])
    # an unmapped tag can't be placed
    assert entry_to_map({"pose": {"tag": 99, "xyz": [0, 0, -1]}}, poses) is None


def _free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def test_telemetry_feed_auto_tracks_non_defenders_only():
    port = _free_port()
    feed = TelemetryFeed(port, None, {10: np.eye(4)}, exclude=["TELLO-2"], tag10_height=1.2)
    try:
        msg = {"v": 1, "seq": 1, "drones": [
            {"name": "TELLO-2", "team": "defense", "fly": True, "h": 1.2, "pose": {"tag": 10, "xyz": [0, 0, -2], "age": 0}},
            {"name": "TELLO-8", "team": "attack", "fly": True, "h": 1.5, "pose": {"tag": 10, "xyz": [1.0, 0.0, -3.0], "age": 0}}]}
        out = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        for _ in range(5):
            out.sendto(json.dumps(msg).encode(), ("127.0.0.1", port))
            time.sleep(0.05)
        t0 = time.time()
        while not feed.tracks() and time.time() - t0 < 2:
            time.sleep(0.05)
        tracks = feed.tracks()
        assert [t.name for t in tracks] == ["TELLO-8"]
        assert np.allclose(tracks[0].pos, [1.0, 1.5 - 1.2, 3.0], atol=1e-6)   # height from h
        assert tracks[0].alive
    finally:
        feed.close()


def test_swarm_bus_feed_reads_formation_poses_from_their_hub():
    from policies.telemetry import TelemetryBus, TelemetryHub     # their code, unmodified
    bus = TelemetryBus()
    hub = TelemetryHub(bus)
    feed = SwarmBusFeed(None, exclude=["TELLO-3"], hub=hub.addr)
    try:
        t0 = time.time()
        while time.time() - t0 < 3 and not feed.tracks():
            bus.publish("formation/drone8", {"state": "APPROACH", "height_cm": 120,
                                             "p": [0.4, 0.0, 2.0], "fix_age": 0.05})
            bus.publish("formation/drone3", {"state": "APPROACH", "height_cm": 120,
                                             "p": [0.0, 0.0, 3.0], "fix_age": 0.05})    # ours: excluded
            time.sleep(0.1)
        tracks = feed.tracks()
        assert len(tracks) == 1 and drone_number(tracks[0].name) == 8
        assert np.allclose(tracks[0].pos, [0.4, 0.0, 2.0])
    finally:
        feed.close()
        hub.close()
