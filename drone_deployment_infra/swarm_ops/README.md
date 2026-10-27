# swarm_ops: Tello swarm commands

All commands run from `drone_deployment_infra/swarm_ops/`. Config lives in `../config/`
(`dongles.json`, `map.yaml`, `tags.yaml`, `camera/tello_<n>.yaml`).

```
cd ~/Action_Space/drone_deployment_infra/swarm_ops
source <your venv>/bin/activate
```

---

## 1. One-time setup (per laptop)

```
pip install -r requirements.txt
echo "net.ipv4.conf.all.rp_filter=2" | sudo tee /etc/sysctl.d/99-tello.conf
sudo sysctl --system
```
- Plug the USB Wi-Fi adapters into a **powered hub** (adapter names in `dongles.json`
  belong to those exact dongles).
- Create every drone's Wi-Fi profile from `dongles.json` (drones can be OFF):
```
python connect.py --create --all
```
  Each profile is locked to its adapter + its drone's radio (BSSID), auto-connects,
  uses static IP `192.168.10.10<n>`, 2.4 GHz only.

**New drone?** Add its entry in `../config/dongles.json` (`iface`, `drone`, `ssid`, `bssid`),
then `python connect.py --create <n>`.
Find a drone's BSSID: `nmcli -f SSID,BSSID dev wifi list | grep TELLO`.

---

## 2. Connect drones (every session)

Turn the drones on, wait ~20-30 s (they join by themselves), then check:
```
python connect.py 4 5 6 7 9          # verify; connects any that aren't up yet
python connect.py --all              # every drone in dongles.json
python connect.py 4 5 6 7 9 --wait 60   # run FIRST, then power on: each joins as it appears
python connect.py 4 5 6 7 9 --down   # disconnect
```
Every drone must say `OK`. `NOT DETECTED` = adapter unplugged / hub unpowered.

Tip: chain it so a flight only starts when all drones are ready:
```
python connect.py 4 5 6 7 9 && python swarm_formation.py --only 4,5,6,7,9 --fly --fast --headless
```

---

## 3. Swarm formation (`swarm_formation.py`)

Drones take off, find tag 10 (or any mapped tag), fly to slots on an arc in front of
tag 10, hold, land. **Without `--fly` it is a dry-run** (no takeoff, no motion).

| Mode | Command |
|---|---|
| Dry-run (props off, check tags/video) | `python swarm_formation.py --only 4,5,6` |
| Normal (slow, careful) | `python swarm_formation.py --only 4,5,6 --fly` |
| Fast | `python swarm_formation.py --only 4,5,6 --fly --fast` |
| Headless (no windows, terminal only) | `python swarm_formation.py --only 4,5,6 --fly --fast --headless` |
| Flip wave (back flip one by one) | `python swarm_formation.py --only 4,5,6 --fly --fast --flip` |
| Recommended | `python swarm_formation.py --only 4,5,6 --fly --fast --headless --rest-radius 0.15 --hold 1` |

**Options**
| Flag | Meaning |
|---|---|
| `--only 4,5,6` | which drones |
| `--fly` | actually fly (without it: dry-run) |
| `--fast` | faster search/approach/turns (same final precision) |
| `--headless` | no windows; keys in terminal: `q`+Enter land all, `e`+Enter EMERGENCY |
| `--no-video` | no camera windows, only the top-view map |
| `--display-hz 5` | refresh windows less often (saves CPU) |
| `--rest-radius 0.15` | slot "close enough" zone in m (default 0.08); inside it the drone just holds |
| `--hold 1` | seconds to hold the formation before landing (default 5) |
| `--flip` | back flip wave once in formation (needs battery >= 55 %, ~1 m free behind each slot) |
| `--radius 1.3` | slot distance from tag 10 in m (don't go below ~1.2) |
| `--slot-offset 0` | slot height vs tag 10 center, cm |
| `--no-map` | use tag 10 only (ignore map.yaml) |
| `--ignore-tags 6,7` | skip specific mapped tags |

**Keys** (window mode): `q` = land all, `SPACE` = EMERGENCY motor stop. `Ctrl+C` = land all.

**Placement:** drones 2-3 m in front of tag 10 (within +-60 deg), 1 m+ apart,
**facing tag 10** (search becomes instant). More drones -> the arc moves further out
(5 drones ~1.5 m, 6 drones ~2 m from tag 10).

**Logs:** `logs/formation_<date>_<time>.csv` (state, position, commands, battery per drone).

---

## 4. Multi hover (`multi_hover.py`, no video)

All selected drones take off together, hover, land. Good for testing many drones.
```
python multi_hover.py --config ../config/dongles.json --only 1,2,3,4,5,8,9 --hover 10
```
All-or-nothing preflight (one failure -> nobody takes off). `Ctrl+C` lands all.
Drones ~1.5 m apart.

---

## 5. Attack + defense, scripted policies (`swarm_policies.py`, umbrella)

Runs one or more policies at once, each on its own drones: `--run <policy>:<drones>`,
repeatable. **Without `--fly` it is a dry-run.** One FLY prompt for everyone,
all-or-nothing preflight (formation drones included), `Ctrl+C` lands all.
Policies live one per file in `policies/` (add one there and register it in `policies/__init__.py`).

| Policy | What it does |
|---|---|
| `spin` | defense dummy: take off, hold the takeoff height, spin in place (`--yaw`, default 40) |
| `levels` | lower-numbered half at 3 ft (91 cm), the rest at 6 ft (183 cm), hover |
| `hover` | hold the takeoff height, or `--height <cm>` (40-250) |
| `formation` | `swarm_formation.py` on those drones, as its own process |

```
python swarm_policies.py --run spin:6,7,9 --run formation:4,5           # dry-run
python swarm_policies.py --run spin:6,7,9 --run formation:4,5 --fly     # defense spins, attack forms up
python swarm_policies.py --run hover:4,5,6,7,9 --height 200 --fly --duration 60   # photo hover at 2 m
python swarm_policies.py --run levels:4,5,6,7,9 --fly
```

**Options**
| Flag | Meaning |
|---|---|
| `--run spin:6,7,9` | a policy and its drones (repeat per group; a drone in one group only) |
| `--fly` | actually fly (without it: dry-run, 15 s) |
| `--duration 60` | land all after this many s (default 20; with `formation`: until it lands) |
| `--height 200` | `hover` height above the floor, cm |
| `--yaw -60` | `spin` speed and direction (-100..100) |
| `--formation-args "--fast --hold 15"` | passed to `swarm_formation.py` |

With `formation`, its camera windows and keys (`q`, `SPACE`) still work, but its
`--headless` terminal keys don't (its stdin is the umbrella): use `Ctrl+C` in the
umbrella's terminal. Formation drones only avoid each other: keep the other groups out
of the space in front of tag 10. Safety: a drone with no status stream for 3 s or
battery < 15 % lands on its own; takeoff needs battery >= 40 %.

**Telemetry between policies** (`policies/telemetry.py`). Any policy can publish live data
and read what the others publish; only the latest value per topic is kept, with its age.
```python
self.bus.publish("spin/drone6", {"yaw_rc": 40})        # in step() or anywhere
hit = self.bus.latest("formation/drone4", max_age=0.5)  # (data, age_s) or None if missing/stale
self.bus.since("formation/")                            # {topic: (data, age_s)}
```
In-process policies get the umbrella's bus as `self.bus`. A policy in its own process uses
`TelemetryClient.from_env()` (same calls, over localhost UDP to the umbrella; a private bus
when run alone). Published automatically every tick:
| Topic | From | Data |
|---|---|---|
| `status/drone<N>` | umbrella, its own drones | `policy`, `height_cm`, `battery`, `rc` |
| `formation/drone<N>` | `swarm_formation.py` | `state`, `p` (m, map frame; `None` = no fix yet), `psi_deg`, `fix_age`, `height_cm`, `rc` |

Steer on telemetry only while it is fresh (`max_age`): a stalled publisher must not steer a drone.

---

## 6. Map the tags (`map_tags.py`, no flying)

Needed whenever tags are added or moved. Hold the drone (props off) so it sees
tag 10 together with each other tag until all are green, then press `q` to save.
```
python connect.py 4
python map_tags.py --drone 4 --tags-list 10,6,7
```
Every tag needs its size in `../config/tags.yaml` (`sizes_mm`, black square in mm).
Check: dry-run shows no `MAP CHECK` warnings.

---

## 7. Troubleshooting

| Symptom | Fix |
|---|---|
| `NOT DETECTED` / `OSError: No such device` | adapter unplugged or hub unpowered: replug, `nmcli dev status \| grep wlx` |
| `drone not broadcasting` | drone off / asleep / flat battery: power it on |
| Drones switch off while connecting | idle auto-off / heat: power on just before flying, connect right away |
| Takeoff failed (height 0) | low battery, hot, tilted surface, or IMU calibration needed (Tello app) |
| Pixelated video with 5+ drones | 2.4 GHz crowding: power drones on ~10 s apart, spread adapters, EDU video lower (see `EDU_MODIF_README.md`) |
| `Map: tags [...]` doesn't match the wall | re-map with `map_tags.py` |
| Wi-Fi name `TELLO-TELLO-6` | in the Tello app type only `6` (it adds `TELLO-`) |
