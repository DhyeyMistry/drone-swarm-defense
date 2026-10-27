# our_defense

The defense system. It lives inside the Action Space kit and uses the kit's code (`core/`, `cal_and_map/`, `swarm_ops/policies/telemetry.py`, `config/`) without changing any of it. Overview, results and GIFs: [the main README](../../README.md).

## Run

**Attackers and defenders in one command:** the organizers' `swarm_policies.py` runner, with our `defend` policy added.

```bash
python3 swarm_run.py --run formation:1,2 --formation-args "--target 6 --fast --hold 60" --run defend:3,4,5 --fly
```

`swarm_run.py` runs their `swarm_ops/swarm_policies.py` unchanged, registering `defend` as an external policy at runtime. Their runner:
1. starts `defend.py` for our drones (dongle numbers), with its telemetry hub in `$SWARM_TELEM`;
2. counts our `[TELLO-N] PASS` lines;
3. asks its single `Type FLY:` question. Our defenders take off the moment you answer FLY.

Ctrl-C or the end of `--duration` lands everyone. Extra `defend.py` flags go in `--defend-args "..."`. Without `--fly`, nothing takes off.

**Our side only:**

```bash
../../.venv/bin/python -u defend.py --fly --defenders TELLO-3,TELLO-4,TELLO-5 --attackers auto
```

Any Python works: `defend.py` relaunches itself with `<repo>/.venv` if the current interpreter lacks the packages. Each defender takes off **as soon as it connects**. `--dry_run` connects, localizes and prints the rc commands without taking off.

**Browser simulation (no drones):**

```bash
../../.venv/bin/python hil_gui.py --attackers TELLO-8,TELLO-9                                # http://127.0.0.1:8790/
../../.venv/bin/python defend.py --sim --defenders TELLO-2,TELLO-3,TELLO-5 --attackers auto --seconds 3600
```

## What happens in a run

1. **Startup:** prints the map, ignored tags, tag 10's height (worked out from the floor tags in the map), the V formation, the table position and the flight box.
2. **Per defender** (`defender_link.py`, its own thread):
   1. join its Wi-Fi stick and wait for `command` ok;
   2. battery check;
   3. **takeoff → `up --climb_cm` → `forward --advance_cm`** toward tag 10;
   4. hand over to the control loop.

   Video is started afterwards and restarted if it fails. One lock per drone keeps its SDK calls from overlapping.
3. **Control loop** (10 Hz planner, about 20 Hz control):
   1. localize every defender (`upright_positional.py`);
   2. read the attackers (`feeds.py`);
   3. plan (`defense_core.py`);
   4. send rc, with safety checks.
4. **Logs to the terminal:**
   - every role change (`ROLE TELLO-3: hold -> ram TELLO-1`);
   - attackers going live or down;
   - contacts;
   - `MAP CHECK` drops;
   - landings with their reason;
   - a status block every `--log_every_s`: per defender its mode, target, position, rc, fix age, height and battery; per attacker its position, velocity and report age.

   Everything is also written to `../recordings/defend_<stamp>.jsonl`.

## Settings (`defend_config.yaml`, overridable by any flag)

| Setting | Default | Meaning |
|---|---|---|
| `defenders` | `TELLO-2, TELLO-3, TELLO-5` | our drones (names from `config/dongles.json`) |
| `attackers` | `auto` | every drone in the feed that isn't ours and isn't team `defense`; or a list (`TELLO-8`, `drone8`, `8`) |
| `fly` | `true` | `--dry_run` overrides |
| `map`, `tags` | `../config/map.yaml`, `../config/tags.yaml` | room map and tag sizes |
| `ignore_tags` | `2,7,11,12,23` | tags moved after mapping. Others that disagree with tag 10 by > 0.5 m are dropped automatically. |
| `tag10_height_m` | `auto` | from the floor tags in the map (fallback 1.22) |
| `site` | `auto` | the table: just behind the V (or behind the starting line with `formation: home`) |
| `formation` | `v` | hold pattern with no live attacker: `v` (in front of `v_tag`, tip toward tag 10), `home` (start spots), `triangle`, `line` |
| `v_tag`, `v_front_m`, `v_width_m`, `v_depth_m` | `23`, `0.8`, `1.2`, `0.6` | V geometry |
| `feed` | `auto` | `swarm` (swarm_policies bus), `gui` (drone_gui UDP `feed_port` 9100), or both |
| `speed_cap` | `0.35` | sticks are capped at 45 (about 0.8–1 m/s real) |
| `max_lead_s`, `latency_s` | `0`, `0` | aim at the attacker's current position; > 0 uses a predicted intercept point |
| `climb_cm`, `advance_cm` | `30`, `50` | after takeoff |
| `first_fix_s`, `lost_timeout_s` | `30`, `30` | no tag: hover at tag height with a widening yaw search, then land |
| `fence_margin` | `0.4` | flight box = area spanned by the tags, pulled in this much |
| `log_every_s` | `1.0` | status block period |

## Files

| File | What |
|---|---|
| `defend.py` | main program: config, startup, control loop, safety, logging |
| `defense_core.py` | strategy: assignment, ram aim / intercept, goalkeeper, formations, flight box |
| `feeds.py` | attacker positions: `swarm_policies` telemetry bus and `drone_gui` telemetry |
| `upright_positional.py` | defender localization: print-orientation proof, drops moved tags |
| `defender_link.py` | per-drone connect, instant-takeoff hand-off, self-restarting video |
| `swarm_run.py` | the organizers' runner with our `defend` policy (one command for both teams) |
| `defend_config.yaml` | settings |
| `hil_gui.py` | browser simulation: drag attackers with the mouse, or let them attack |
| `sim_benchmark.py` | many simulated attack rounds, writes a CSV and a chart |
| `feed_check.py` | print the attacker feed in map coordinates (no flying) |
| `axis_check.py` | measure one drone's real motion per stick axis against the tags |
| `dummy_attacker.py`, `duel.py` | practice attacker and a one-command real duel |
| `make_report.py` | HTML report (charts, video frames, run table) from the logs |
| `home/` | single-tag home test: `map.yaml`, `tags.yaml` (set your printed size), `tag10_print.png` |
| `tests/` | `pytest` suite (also run in CI) |

## Tests

```bash
cd drone_deployment_infra/our_defense
../../.venv/bin/python -m pytest tests          # with ROS sourced, add: -p no:launch_testing -p no:launch_ros
```
