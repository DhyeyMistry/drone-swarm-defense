# drone_deployment_infra

Real-world drone driving: calibrate the cameras, map the room in the tag-10 frame,
localize the drone, connect the fleet, and run track-driving policies. (The Isaac Lab
simulation lives in `../simulation_infra/`.)

All real-world Python runs under the repo venv `../.venv`. The calibration/mapping
package is editable-installed from `cal_and_map/`:

```bash
../.venv/bin/pip install -e cal_and_map
```

## Layout

| Folder | What |
|---|---|
| `cal_and_map/` | Camera (focal-length) calibration + AprilTag room mapping + localization. See its README and `calibrate_map.md`. |
| `config/` | All configuration: tags to map + sizes, camera focal lengths, dongle→drone map, fleet roles. See `config/README.md`. |
| `core/` | Infrastructure: connect the quick-connect drones (`tello_link.py`), the flight engine (`tello_dual_video.py`), the positional lookup (`positional.py`), flight-track viz. See `core/README.md`. |
| `drone_deployment_tests/` | The policies and tests we fly down the track. See its README. |
| `swarm_ops/` | Multi-drone flying: connect the fleet (`connect.py`), swarm formation, multi hover, tag mapping, and scripted attack/defense policies run together (`swarm_policies.py`, one policy per file in `policies/`). See its README. |
| `recordings/` | All flight recordings (`<session>_<drone>_<stamp>.avi`) and reports. |
| `outputs/` | Per-flight live tracks, training logs, and the flight visualization. |

## Flow

1. **Calibrate** each drone's camera → `config/camera/<drone>.yaml` (focal length).
2. **Map** the room → `config/map.yaml` (every tag in tag-10's frame).
3. **Connect** the quick-connect drones with `core/tello_link.py`.
4. **Fly** a policy from `drone_deployment_tests/`; `core/positional.py` turns any mapped
   tag in frame into the drone's position relative to origin.

Swarm flights (several drones, attack + defense): `swarm_ops/README.md`.

Top-level command references: `calibrate_map.md` (calibrate + map) and `FLYING.md` (fly).
