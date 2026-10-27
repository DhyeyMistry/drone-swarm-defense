# drone_gui

Browser GUI for flying several Tellos at once, one per USB WiFi dongle.

```bash
cd drone_deployment_infra/drone_gui
../../.venv/bin/python -u drone_gui.py        # open http://127.0.0.1:8780/
../../.venv/bin/python -u drone_gui.py --sim 8 --port 8781   # 8 simulated drones, no hardware
```

Ctrl-C lands anything airborne and quits. In `--sim` mode the top bar also has a
**Stop sim** button (click twice) that shuts the simulator down. Don't run it at the same time as
`core/tello_link.py` / `tello_dual_video.py` (they bind the same ports).

## What it shows

- Top bar: counts of connected, flying, seeing their target tag, and dongles (or sims),
  plus the fleet buttons.
- **Cameras**: 4 columns x 2 rows (more rows past 8 drones; empty slots are greyed). Each
  tile shows its slot number, name, battery, mode and whether it sees its target tag
  (green outline = target tag, yellow = other tags; yellow border = airborne).
- **Controls**: one panel per drone in the same numbered order: link, battery, height,
  tags, last message, WiFi connect, target tag, standoff, flight buttons, stick pad.
- 2 columns on narrow windows, 1 on phones.
- **Telemetry** box in the top bar: ON/off, rate in Hz, destinations (`host:port, ...`)
  and the live send rate, datagram size and sequence number. Changes apply at once.
  Each tile and panel shows its team (ATTACK red / DEFENSE blue); the panel's Team
  selector overrides it. Format, teams and a listener: [../drone_telemetry/](../drone_telemetry/README.md).

## Controls

| Control | What |
|---|---|
| Connect / Disconnect | Join the drone's WiFi on that dongle (`nmcli`), SDK handshake, start video. |
| Target tag, Land at (m) | Which AprilTag this drone flies to, and how far in front of it to land. |
| **Auto: tag → land** | Take off if needed, face the target tag, close to the standoff, land. |
| Take off / Hover / Land | Manual flight basics. |
| Click a tile / panel, or `1`-`8` | Select that drone for the keyboard. |
| Pad / keyboard | Manual sticks on the selected drone: `W/S` fwd/back, `A/D` left/right, `↑/↓` up/down, `Q/E` yaw. Deadman: release and it hovers. |
| **Connect all** | Connect every assigned drone that isn't linked yet. |
| **Auto all** | Auto on every linked drone with a target tag, takeoffs 2 s apart. |
| **LAND ALL** / `Space` | Land every drone. |
| Motor stop (click twice) | `emergency`: motors cut, the drone **falls**. Last resort. |

## Flying two drones to their own tags

1. Plug in both dongles and turn both drones on. Each dongle must be assigned to a drone
   in `../config/dongles.json` (`../core/tello_dongle_setup.py`).
2. Click **Connect** on both cards and check that each camera view is live.
3. Set each card's **Target tag** to a different tag (defaults: TELLO-2 → 8,
   TELLO-FE2950 → 9, see `DEFAULT_TAGS` in `drone_gui.py`). Aim each drone so its tag
   is visible and turns green.
4. Click **Auto all** (or Auto on each card). Each drone flies to its own tag and
   lands `Land at` metres in front of it.

Each drone uses only the tag in its own camera, not `../config/map.yaml`, so this works in
any room. Paths are not checked against each other, so set the drones up so they
can't cross.

## Testing many drones without hardware: `--sim N`

`--sim N` replaces the dongles with N simulated Tellos (`sim_drone.py`). Each one stands
2-3 m in front of its own AprilTag (sim i -> tag i, tags 1-9 repeat), slightly off-axis
and turned. It renders what its camera would see through the real camera matrix. The
**same** detection, Auto controller and safety code as a real drone run on those frames.
Use it to check the grid, Connect all / Auto all / LAND ALL, the takeoff stagger, manual
keys, and the lost-tag search. **Reset pose** (sim only) puts a landed drone at a new
random start. It runs no sockets, so it can run beside the real server on another `--port`.

With real hardware, every extra drone needs its own USB WiFi dongle assigned in
`../config/dongles.json` (`../core/tello_dongle_setup.py`). It shows up in the next free
slot within ~3 s of being plugged in.

## How Auto steers

It yaws to keep the target tag centred, flies forward/back until the tag is `Land at` m
away, climbs or sinks if the tag drifts off the top or bottom of the frame, and lands once
it is facing the tag within the distance tolerance. It does **not** use the tag's estimated
orientation (`tello_policy.track_rc`). For a 13 cm tag a few metres away, that estimate
flips between two mirror-image solutions. In sim, a drone 0.4 m to the side read as centred
with a 9° heading error, and `track_rc` drifted away instead of converging. So the drone
lands facing the tag at the right distance, but not necessarily on the tag's centre line.

## Safety rules built in (`tello_unit.py`)

- No takeoff below 20% battery; auto-land below 10%.
- Auto: tag lost → hover, then slow yaw sweep after 1 s; lands where it is after 10 s lost;
  lands after 60 s airborne.
- Link dropped mid-flight → lands as soon as the link is back.
- rc is streamed at 20 Hz while airborne (also keeps the Tello from its 15 s auto-land).
- Server listens on 127.0.0.1 only.

Drones without a calibration in `../config/camera/` use TELLO-3's camera numbers
(card shows "provisional camera"); distances are then approximate.
