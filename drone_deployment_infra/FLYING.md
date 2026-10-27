# Flying the real Tellos

Tools for flying physical DJI Tellos over USB WiFi dongles, approaching AprilTag 10,
and reviewing flights. Separate from the Isaac Sim kit in `README.md`.

All paths are relative to the repo root (`~/action-space`). The three `./` scripts are
executables (venv shebang); the rest run under the venv Python.

## 1. Number the dongles (once per rig)

Each USB WiFi stick is identified by its interface name `wlx<mac>`. Record which
numbered stick flies which drone into `dongles.json` (committed, so the rig is shared).

```bash
./tello_dongle_setup.py            # plug ONE stick in at a time, Enter to capture, 'done' to finish
./tello_dongle_setup.py --watch    # background: auto-number each newly plugged stick
./tello_dongle_setup.py --list     # print the current dongle -> drone table
```

Set a stick's drone (e.g. #3 -> TELLO-3) during capture, or edit `dongles.json`.
Video port is `17000 + number`.

### Lock each NetworkManager profile to its stick

A drone's WiFi profile must be pinned to its numbered stick's interface:

```bash
nmcli connection modify TELLO-1 connection.interface-name wlx6c4cbce345b3
nmcli connection modify TELLO-3 connection.interface-name wlx6c4cbce33375
nmcli connection modify TELLO-4 connection.interface-name wlx58d8125ecf0d
```

(Use `./tello_dongle_setup.py --list` for the current interfaces.)

## 2. Fly — the subroutine

`tello_link.py` is the one command: it finds whichever numbered sticks are plugged in,
joins each drone's WiFi, launches the flight engine for exactly those drones, and flies
them. It reconnects on signal loss and, if the link drops in flight, **lands on
reconnect** (no re-takeoff). The drone takes off ~20 s after start.

```bash
./tello_link.py            # connect present sticks and FLY toward tag 10, land at 1.3 m
./tello_link.py --no_fly   # connect + stream only, NO takeoff
```

Live camera + pose while flying: http://127.0.0.1:8770/

## 3. Walk-range test (how far before the link drops)

Put the laptop at tag 10's X,Y. The drone's measured distance to tag 10 is then your
distance from the base. The drone never takes off.

```bash
./tello_range_test.py              # first plugged-in drone
./tello_range_test.py TELLO-1      # a specific drone
```

Carry the drone away keeping tag 10 in the camera; Ctrl-C prints the max range reached
while connected (metres and feet).

## 3b. Hover link test (when/if it drops in flight, and reconnect time)

Takes off, holds position (no approach), and probes the control link ~3x/s to log
exactly when it disconnects, whether it recovers, and how long the reconnect takes.
Auto-lands after `--seconds` (default 60) or on Ctrl-C.

```bash
./tello_hover_test.py                   # first plugged-in drone, 60 s
./tello_hover_test.py TELLO-1 --seconds 90
```

Writes `outputs/hover_<stamp>.jsonl`. A drop shown with `wifi up` means the WiFi stayed
associated and only the drone's SDK channel went quiet (motor-EMI signature), not a
software teardown.

## 4. The flight engine directly (optional)

`tello_link.py` normally launches this for you. To run it standalone:

```bash
.venv/bin/python -u tello_dual_video.py             # fly whichever sticks are plugged in
.venv/bin/python -u tello_dual_video.py --no_fly TELLO-1   # no takeoff, one drone
```

Each flight writes everything into one place, `recordings/`:
- a recording `recordings/<session>_<drone>_<stamp>.avi`,
- a live track `recordings/<session>_<drone>.live.jsonl` (time, wifi, pose),
- a **training log** `recordings/<session>_<drone>.train.jsonl` -- one row per ~20 Hz tick
  with the action and the drone's response, for fitting an action -> motion MLP:
  `{t, rc:[right,forward,up,yaw], vel:{r,f,u,yaw}, h, tag:{x,y,z,range}, mode, fresh}`.

Flight behavior: it tracks the tag **closed-loop** (snaps the estimate to every fresh
sighting, odometry only bridges gaps). If it loses the tag for `SEARCH_AFTER_S`, it
**stops and sweeps yaw** left then right (`search` mode) to re-find it rather than
flying on odometry. It approaches to `LAND_RANGE_M` and lands when sighted + centered.

## 5. Visualize flights

```bash
.venv/bin/python scripts/extract_track.py      # reconstruct a track for every recording lacking one
.venv/bin/python scripts/build_flight_viz.py   # bake ALL tracks into recordings/flight_viz.html
xdg-open recordings/flight_viz.html
```

The page shows, per flight: a top-down path (where the drone thought it was in the
tag-10 frame), a "saw an AprilTag" lane, a "tag-10 pose" lane, and a **WiFi lane**
(green connected / red dropped) — the WiFi lane fills only for live tracks.

## Tunables

| Knob | File | Default | Meaning |
|---|---|---|---|
| `LAND_RANGE_M` | `tello_policy.py` | 1.3 | approach and land this far from the tag |
| `YAW_PID`, `LATERAL_PID`, `DIST_PID` | `tello_policy.py` | see file | `(kp, ki, kd)` per axis; I off (odometry drift) |
| `LAND_TOL_M` | `tello_policy.py` | 0.12 | land when within this of the target |
| `LAND_ON_LOSS` | `tello_dual_video.py` | `True` | in-flight link drop -> land on reconnect (False = resume) |
| `SEARCH_AFTER_S` | `tello_dual_video.py` | 0.4 | hover+sweep to re-find the tag after this long unseen |
| `SEARCH_YAW`, `SEARCH_LEG_S` | `tello_dual_video.py` | 25, 2.0 | search yaw rate; seconds per left/right sweep leg |
| `VIDEO_SETUP` | `tello_dual_video.py` | `()` | low-bitrate video cmds; empty — this Tello (non-EDU) rejects them |
| `PORT_BASE` | `tello_dongle_setup.py` | 17000 | video port = base + dongle number |

## Known issue: link drops in flight

On this (non-EDU) Tello the control link is solid when the drone is carried (tested to
~6 m) but drops within ~2 s of the motors spinning — **motor EMI** desensitizing the
2.4 GHz radio. The SDK video-shrink commands (`setresolution`/`setfps`/`setbitrate`) are
rejected by non-EDU firmware, so the 720p stream can't be trimmed in software. The
fail-safe (`LAND_ON_LOSS`) lands the drone on reconnect. Hardware fixes: a USB extension
to reposition the dongle toward the flight area, an external-antenna adapter, or a Tello
EDU (which accepts the video commands — then set `VIDEO_SETUP` back).

## Emergency stop

Catch the drone, flip it upside-down (motors cut), or press its power button. In
software, Ctrl-C / kill the subroutine; with no commands the Tello auto-lands after ~15 s.

```bash
pkill -f tello_link.py ; pkill -f tello_dual_video.py
```
