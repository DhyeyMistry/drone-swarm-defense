# core

Infrastructure that runs underneath any policy: connect the drones, stream + track,
localize against the map, and review flights. Run everything with the repo venv
(`../../.venv/bin/python`); `tello_link.py` and the tests are also executable
(`./tello_link.py`).

| File | What | Run |
|---|---|---|
| `tello_link.py` | The autoconnect-and-fly subroutine. Reaps stale engines, joins whichever quick-connect dongles are plugged in, launches the flight engine for them, and kills it on exit. | `./tello_link.py` (fly) · `./tello_link.py --no_fly` (connect + stream only) |
| `tello_dual_video.py` | The flight engine: per-drone video, AprilTag pose, PID approach, land, link-loss handling, and the live tracks / training logs. Normally launched by `tello_link.py`. | `.../python tello_dual_video.py [TELLO-3 ...]` |
| `tello_dongle_setup.py` | Build/inspect the dongle→drone registry `config/dongles.json`. | `./tello_dongle_setup.py` · `--watch` · `--list` |
| `tello_tags.py` | Tag sizes + the tag-10 world-frame pose math used by the engine. | (library) |
| `positional.py` | "Where is the drone?" — loads `config/map.yaml` + a drone's camera calib and returns the drone's position in the tag-10 frame from any mapped tag in a frame. | `Positioner("tello_3").locate(frame)` |
| `fleet.py` | Loads `config/fleet.yaml` (roles, quick-connect, count). | `.../python fleet.py` prints it |
| `extract_track.py` | Reconstruct a per-flight track from a recording (tag-10 pose over time). | `.../python extract_track.py` |
| `build_flight_viz.py` | Bake all tracks into `recordings/flight_viz.html`. | `.../python build_flight_viz.py` then open the html |

Outputs land in `../recordings/` (video) and `../outputs/` (tracks, training logs, viz).
Policies to drive the track live in `../drone_deployment_tests/`; the engine imports the
active one from there.
