# drone_deployment_tests

The policies we're developing to drive the drone down the track, plus the link/flight
tests used to validate them. The flight engine (`core/tello_dual_video.py`) imports the
active policy from here.

Run with the repo venv; the test scripts are executable (venv shebang), so `./name.py`
works, or `../../.venv/bin/python <script>`.

| File | What | Run |
|---|---|---|
| `tello_policy.py` | The current track-driving policy: per-axis **PID** (yaw / lateral / distance), yaw-first alignment, approach to `LAND_RANGE_M` (1.3 m), and the search-on-tag-loss behavior. Tunable gains at the top. Imported by the engine. | (library — flown via `core/tello_link.py`) |
| `tello_range_test.py` | Walk-range test: carry the drone (no takeoff) away from the base at tag 10 and read the max range it stays connected, using the tag-10 distance as the ruler. | `./tello_range_test.py [TELLO-3]` |
| `tello_hover_test.py` | Link test under flight: takes off and holds (or flies a forward/back pattern), streaming rc steadily with passive state-based link detection; logs disconnect/reconnect timing. | `./tello_hover_test.py [--rate 20] [--speed 30 --leg 3] [--seconds 60]` |
| `tello_video_check.py` | Video-path diagnostic (no takeoff): connects the quick-connect drone(s), `streamon`, counts raw H.264 bytes on UDP 11111, then tries to decode. Run it when a flight logs `decode open failed`. | `./tello_video_check.py [--seconds 8]` |
| `tello_to_origin.py` | Fly the quick-connect drone(s) to the origin (tag 10) using the map (`core/positional.py`, works from ANY mapped tag) and land at `--range` (1.5 m). One present -> one flies; two -> both, height-separated. Records commands. | `./tello_to_origin.py [--range 1.5]` |
| `tello_swarm_to_origin.py` | The quick-connect drones cross the room to their OWN landing spots 1.5 m from tag 10, spread 0.5 m apart, each at its own cruise height so paths don't cross. Map-based per drone; records every rc command + pose to `../recordings/` for the simulation. | `./tello_swarm_to_origin.py [--seconds 50]` |

New policies go here as separate modules; point the engine at the one you want to fly.
Everything a flight produces lands in one place, `../recordings/`: the video
(`<session>_<drone>_<stamp>.avi`), the action+pose training log for sim replay
(`<session>_<drone>.train.jsonl`), and the visualization track (`.live.jsonl` live, or
`.track.jsonl` reconstructed by `core/extract_track.py`). No hunting between folders.

**Which drones a test flies comes from one place: `config/fleet.yaml` `quick_connect`.**
The shared connection subroutine `core/test_link.py` connects to whichever of those
drones have their USB stick plugged in (one listed and present → one flies, two → both),
over the engine's stick-pinned sockets, and waits for every one before any takes off.
Set `quick_connect` to the drones you want a test to use. Each drone needs a dongle in
`config/dongles.json` (`core/tello_dongle_setup.py`) and a camera calib in
`config/camera/<drone>.yaml` (falls back to `camera/default.yaml` if uncalibrated).
Test-specific layout (landing spots, cruise heights) lives in the test script, not a
config file.

Safety: these command real motors. Keep the area clear and be ready to catch the drone;
`tello_link.py` lands on signal loss and won't re-launch mid-flight.
