# cal_and_map

Calibrate a Tello's camera, map AprilTags (tag36h11) in the **tag-10 origin frame**, and
localize the drone against that map. Standard DJI/Ryze Tello (SDK 1.3, not EDU).

Run with the repo venv (`../../.venv/bin/python`); the `tello_*` scripts have a venv
shebang, so `./tello_map.py` also works. Config is read from `../config/`
(`tags.yaml`, `camera/<drone>.yaml`, `fleet.yaml`, `dongles.json`, `map.yaml`). Per-drone
calibration identity (iface, ssid, bssid, serial) lives in `fleet.yaml`'s `drones:`
section. The scripts bring up the drone's WiFi themselves (NetworkManager profile per
that identity); there is no manual connect step.

## Safety rule

The mapping/calibration comms only ever send a read-only allowlist
(`command streamon streamoff battery? sdk? sn? wifi? time?`). Any flight command
(takeoff, land, rc, ...) raises `CommandNotAllowedError` before a socket call — these
tools never fly the drone.

## The three tools

### `tello_apriltag.py` — see the tags
Live Tello video with tag36h11 detection and the camera's pose relative to a single
on-screen tag (x/y/z of the camera in that tag's frame). Use it to confirm tags are
detected and the pose looks sane before mapping.
```
python tello_apriltag.py --tag-size 0.13        # black-square side, metres
```

### `tello_map.py` — build the map
Connects to the first `quick_connect` drone in `fleet.yaml` (override `--drone TELLO-3`),
and maps tags **10, 5, 6, 7** into the tag-10 frame: hold the drone so it sees tag 10
together with the others (two tags in a frame link them; a chain 10↔6↔5 also works).
Green = placed, yellow = seen-not-yet-linked. **`q`/`s`/`Esc` in the window or `Ctrl+C`
in the terminal saves** `../config/map.yaml` (terminal prints progress each second).
```
./tello_map.py                                  # or --drone TELLO-3 --tags-list 10,5,6,7
```

### `tello_calibrate.py` — map from a recorded video (offline)
Same map, built offline from one walkaround video instead of a live drone. The video
must show tag 10 sharing a view with each other tag (directly or via a chain).
```
python tello_calibrate.py ../recordings/walk.avi
```

## Camera calibration (focal length)

The camera matrix / distortion (focal length = `camera_matrix[0][0]`) is produced by the
ChArUco flow and stored in `../config/camera/<drone>.yaml`:
```
python scripts/02_make_charuco.py                       # print the board, measure a square
python scripts/03_calibrate.py --drone tello_3 --square-mm 41.2
```

## Localization

- `scripts/06_localize.py --drone tello_3` — live: drone position in the tag-10 frame
  from any mapped tag in view.
- `core/positional.py` (`Positioner(drone).locate(frame)`) — the library the flight
  engine calls: image + `map.yaml` → drone position relative to origin, from any mapped tag.

## Layout

| Path | What |
|---|---|
| `tello_apriltag.py` · `tello_map.py` · `tello_calibrate.py` | the three tools above |
| `scripts/0{1..6}_*.py` | numbered helpers: connect/view, charuco, calibrate, tag-check, map, localize |
| `tello_calibration/` | the package (detection, calibration, tag-map math); editable-installed |
| `tello_calibration/tags/mapping.py` | map math; `python -m tello_calibration.tags.mapping` self-checks |
