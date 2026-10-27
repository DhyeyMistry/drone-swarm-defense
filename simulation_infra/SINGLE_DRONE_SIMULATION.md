# Single-drone simulation: train, tune, watch

The guide is written for the single-drone task (`AS-Tello-Room-Single-v0`); every command
also runs the 4-drone task with `--task AS-Tello-Room-v0`.

Two Isaac Lab tasks fly simulated DJI Tellos in the real mapped room. In both, a drone's
**only** position source is its AprilTag sightings, the same as `core/positional.py` on the laptop.

| Task | Drones | Start | Goal | Use it for |
|---|---|---|---|---|
| `AS-Tello-Room-Single-v0` | 1 | on the floor, then the SDK `takeoff` | land on one of the 4 spots (random) | **tuning rewards** (fast to learn, easy to read) |
| `AS-Tello-Room-v0` | 4, one central controller | hovering (takeoff already done) | each lands on its own spot, no contact | the full formation |

Both start inside a 2 m × 2 m box, 6–7 m from tag 10 (the four drones at least 0.2 m apart,
`spawn_min_separation`). Each drone faces tag 10 ±15°, or, 30% of the time
(`spawn_search_prob`), starts turned 75–180° away with no tag in view and has to turn to find
one (all the mapped tags are at the tag-10 end of the room). The landing spots are on the floor
1 m in front of the tag-10 wall, 0.5 m apart. Each spot has a **waypoint 1.22 m above it**
(radius 0.3 m). Entering it counts only at or below 0.3 m/s (`waypoint_max_speed`, the Kalman
filter's speed estimate, judged once the filter's velocity is settled: `speed_trust_std`).
New runs use reward version 2: fast entries can recover, progress uses discounted potential
shaping, and accurate touchdown is the main reward. Reaching the waypoint does not by itself
authorize landing: the current estimated pose, speed, uncertainty and freshness must pass
the shared landing gate continuously for 0.5 seconds.

---

## 1. Shell setup (every new terminal)

Isaac Lab lives in a uv virtual environment, **not** a conda env. Just the environment:

```bash
source ~/isaac/env_isaaclab/bin/activate
```

The full setup, for every new terminal:

```bash
cd ~/Action_Space/simulation_infra && unset CONDA_PREFIX CONDA_DEFAULT_ENV && source ~/isaac/env_isaaclab/bin/activate && export OMNI_KIT_ACCEPT_EULA=YES TELLO_USD_PATH=$HOME/isaac/assets/tello.usd && mkdir -p outputs
```

Every command below assumes this shell, in `simulation_infra/`. Always run Python with `-u`
and send output to a log under `outputs/`: Isaac Sim prints hundreds of harmless warnings,
and it can drop buffered output when it exits. Conda's `(base)` can stay active, but the
`unset` is required: otherwise Isaac Lab picks conda's python.

When using an isolated checkout, ensure its source wins over any editable installation:

```bash
export PYTHONPATH="$PWD/source/action_space_kit${PYTHONPATH:+:$PYTHONPATH}"
```

---

## 2. Check the environment first (~3 min)

```bash
python -u scripts/check_room_env.py --headless --task AS-Tello-Room-Single-v0 > outputs/check_room_single.log 2>&1
grep -E "^\s*\[(ok|FAIL)|^== |zero |PASSED|FAILED" outputs/check_room_single.log
```

Pass: all `[ok]` lines and `ALL CHECKS PASSED`.
To run only the Kalman filter checks: add `--test kalman`.

The policy-specific checks accept any saved single-drone checkpoint; the integration check
temporarily uses two-second episodes to exercise timeout/reset behavior:

```bash
python -u tests/test_policy_contract.py > outputs/policy_contract_tests.log 2>&1
python -u scripts/check_policy_pipeline.py --headless --checkpoint <run>/model.zip > outputs/policy_pipeline.log 2>&1
python -u tests/check_export_replay.py --checkpoint <run>/model.zip > outputs/export_replay.log 2>&1
```

These check terminal observations through normalization/frame stacking, exact episode counts,
landing-gate failures, four controlled touchdowns, and simulator/adapter/export inference parity.

---

## 3. Tune the rewards

### Where they are

The reward weights are in
`source/action_space_kit/action_space_kit/tasks/direct/tello_room/tello_room_env_cfg.py`
(the `# -- rewards` block). The single-drone task inherits them. All are per drone, per policy step (74 ms):

| Field | Default | Logged as | What it does |
|---|---|---|---|
| `rew_potential_scale` | `40.0` | `closer_reward` | bounded potential toward waypoint, then descent; `gamma * Phi(next) - Phi(previous)`, zero next potential on true termination |
| `reward_gamma` | `0.995` | — | must equal PPO gamma; timeouts retain their final-state potential and bootstrap |
| `rew_waypoint` | `25.0` | `waypoint_reward` | once on slow waypoint entry; does not open the landing gate by itself |
| `rew_settle_position` / `rew_settle_speed` | `-2.0` / `-1.0` | `settling_loss` | per-second position and speed costs after waypoint entry, before landing |
| `rew_slowdown` | `0.0` | `slowdown_reward` | legacy new-low-speed term, disabled by default |
| `rew_time` | `-0.5` | `landing_loss` | every step until landed on the spot; ending early (crash, landing off the spot) pays the remaining steps |
| `rew_no_fix` | `-0.5` | `no_tag_loss` | every step the drone flies with no tag fix |
| `rew_violation` | `-100.0` | `crash_loss` | crash, wall, ceiling, flipping |
| `rew_collision` | `-1000.0` | `collision_loss` | each drone within 0.25 m of another (4-drone task); ends the episode |
| `rew_early_land` | `-5.0` | `early_land_loss` | confirmed requests before the waypoint; afterward blocked requests wait for the gate without a repeated penalty |
| `rew_land` | `1000.0` | `land_reward` | only touchdown within the assigned spot's success radius |
| `rew_all_landed` | `200.0` | `formation_reward` | success bonus: every drone landed within `success_radius` of its spot |
| `rew_scale_separation` | `-20.0` | `separation_loss` | per second, closer than 0.5 m to another drone (always 0 with one drone) |
| `rew_scale_action_rate` | `-0.05` | `action_rate_loss` | per second, jerky sticks |

Geometry that shapes the rewards:

| Field | Default | Meaning |
|---|---|---|
| `waypoint_height` | `1.22` m | aligned with the measured tag center; the old 1.5 m target clips the large tag in the camera view |
| `waypoint_radius` | `0.3` m | slow entry counts as reached; settling guidance continues afterward |
| `success_radius` | `0.3` m | touchdown this close to the spot counts as on the spot |
| `episode_length_s` | `75` s | episode cap |
| `takeoff_height` | `(0.8, 1.0)` m | where the firmware takeoff stops (single-drone task) |

The formulas themselves are in `_get_rewards()` in `tello_room_env.py`.

### Two ways to change them

**Per run, from the command line (recommended for tuning):** append `env.<field>=<value>`
to the train command. The code is untouched, and the values used are saved with the run in
`logs/sb3/<task>/<run>/params/env.yaml`.

```bash
... env.rew_potential_scale=30.0 env.rew_no_fix=-0.25 env.rew_land=1000.0
```

**Permanently:**
- **Single-drone task only:** add the field to the class in `tello_room_single_env_cfg.py`
  (e.g. `rew_potential_scale = 30.0`).
- **Both tasks:** edit the defaults in `tello_room_env_cfg.py`.

### Reading the result

Each term gets its own graph, `Rewards/avg_<term>_per_episode` (averaged per drone, per
episode), so you can see which term dominates. Outcomes are under `Episode/`:

- `success_rate`
- `drones_reached_waypoint`
- `waypoint_reached_pct`: % of drones that got inside their waypoint, per episode
- `drones_landed`
- `drones_landed_on_spot`
- `spot_err_m`
- `crash_rate`
- `duration_s`
- `tag_fix_frac`
- `tag_losses`

---

## 4. Train

### Headless (full speed: use this for real runs)

```bash
setsid nohup python -u scripts/sb3/train.py --task AS-Tello-Room-Single-v0 --headless --num_envs 512 --max_iterations 1500 \
  --curriculum landing env.auto_land=True env.episode_length_s=15.0 \
  > outputs/train_room_single.log 2>&1 &
```

Advance through `--curriculum landing`, `approach`, then `full`, evaluating each stage before
advancing. All three keep the same 25 observations and five actions. Landing starts hovering
within 10 cm per horizontal axis of its assigned pad (`spawn_near_target_radius`); approach
restores takeoff nearer the target; full restores the 6–7 m start. Monitor
`Episode/land_gate_ready_frac` to distinguish failure to settle from failure to request land.
`--checkpoint` resumes both PPO and its matching normalization statistics and transition count.
`--max_iterations` specifies additional iterations on resume.

The command above uses gated auto-land: PPO flies the RC channels, and the controller sends
SDK land only when the current-state gate passes. This avoids depending on an undertrained
fifth action. Use `env.auto_land=False` to research policy-requested landing, and qualify that
mode independently. The saved checkpoint config records the chosen mode.

The objective is the complete mission from the 6–7 m start, including learned approach,
braking, tag retention/reacquisition, and arrival at the assigned pad. A landing-stage score
does not qualify that mission. When a checkpoint already navigates the room, retain that
behavior by fine-tuning on full starts with shorter episodes mixed in:

```bash
python -u scripts/sb3/train.py --task AS-Tello-Room-Single-v0 --headless --num_envs 512 \
  --checkpoint <full-room-checkpoint>/model.zip --config-source current --curriculum full \
  --max_iterations 40 --save_interval 10 env.auto_land=True \
  env.spawn_short_fraction=0.30 env.spawn_hover_fraction=0.15 agent.learning_rate=0.0001 \
  > outputs/train_full_mission.log 2>&1
```

Fractions apply to episode resets: 55% keep the full start, 30% take off from X=1.5–5.5 m,
and 15% start hovering near their assigned pad. This changes only initial conditions;
navigation and recovery remain learned. Per-episode records include start mode and position.
Resume with saved config after the initial transition to the corrected task. Full/dropout
evaluation suites explicitly disable both replay fractions, so easy starts cannot inflate
the full-mission score. For a direct `play.py` evaluation of a mixed-start checkpoint, add
`env.spawn_short_fraction=0.0 env.spawn_hover_fraction=0.0`.

For a controlled fine-tuning experiment, `--action-std 0.2` reduces Gaussian exploration
on resume without changing the deterministic actor or value predictions. It also clears
the optimizer moments for that exploration parameter. `agent.freeze_normalization=True`
keeps the matching checkpoint's input and reward statistics fixed; this requires a
checkpoint. Both are optional, recorded in the run, and need full-mission evaluation
against the parent checkpoint before promotion. They do not add a navigation controller.

When switching a resumed run from raw rewards to `agent.normalize_value=True`, the
critic must use the new reward units too. With frozen normalization and unclipped rewards,
pass `--value-scale <1 / sqrt(ret_rms.var + epsilon)>`, using the matching VecNormalize
checkpoint's return variance. This scales the value head and clears its critic optimizer
moments while preserving actor outputs. Record the exact factor with the run; evaluation
always reports the original physical success metrics, with reward normalization disabled.

An experimental continuous settling reward is available with
`env.rew_precision_potential=80.0 env.rew_settle_position=0.0 env.rew_settle_speed=0.0`.
It adds a bounded position/speed potential before and after the waypoint, rather than
introducing a new per-step liability when the waypoint flag latches. The precision term
peaks at 80 at the waypoint with zero horizontal speed, with 0.20 m position and 0.15 m/s
speed scales. It uses the same discounted potential difference and absorbing-terminal
handling as the approach term. This changes training rewards only; the landing gate and
success radius stay unchanged. The default weight is zero to preserve saved-run replay.

For a policy that completes missions on only one side of the central tag, the experimental
`collect_room_demonstrations.py` / `imitate_room.py` workflow collects complete successful
flights and trains the same actor on original and reflected examples. Reflection changes
world Y, heading/bearing sine, lateral goal offsets, and lateral/yaw commands. Normalization
is applied after reflecting raw histories; padding remains zero. Validation splits whole
trajectories, keeping each original/reflected pair together. The critic is retained and
PPO actor optimizer moments are cleared before saving. No mirroring or policy routing
runs at inference time.

This is supervised imitation, not an on-policy PPO update on invented transitions. The
room walls and auxiliary tags are asymmetric, so reflection is only a training prior;
every candidate must be evaluated in the original room. The approach is motivated by
[research on task symmetry in robot learning](https://arxiv.org/abs/2403.04359), not by a
claim that this room is a perfectly symmetric MDP.

For example, after the landing stage passes its evaluations:

```bash
setsid nohup python -u scripts/sb3/train.py --task AS-Tello-Room-Single-v0 --headless --num_envs 512 --max_iterations 1500 \
  --checkpoint <landing-run>/model.zip --curriculum approach env.episode_length_s=75.0 \
  > outputs/train_room_single_approach.log 2>&1 &
```

- The run keeps going after you close the terminal.
- Output: `logs/sb3/AS-Tello-Room-Single-v0/<timestamp>/`. Checkpoints are saved every 50
  iterations (`model_<steps>_steps.zip` + `model_vecnormalize_<steps>_steps.pkl`), plus
  `model.zip` at the end.
- To follow it: `grep -E "success_rate|drones_landed_on_spot|iterations" outputs/train_room_single.log | tail`
- To stop it: `pkill -TERM -f "[A]S-Tello-Room-Single"`
- To resume from a checkpoint: add `--checkpoint logs/sb3/AS-Tello-Room-Single-v0/<run>/model_<steps>_steps.zip`

For the 4-drone task, use `--task AS-Tello-Room-v0`. It takes about 27 s per iteration at 512 envs, so 1500 iterations is about 11 h.

### With or without the Kalman filter

The position the policy sees is, by default, the **Kalman-filtered** estimate from
`drone_deployment_infra/core/positional_kalman_filter.py`, the same file the laptop uses.
`env.kalman.enabled` switches it; the tag noise is the same either way. The setting is saved
with each run in `logs/sb3/<task>/<run>/params/env.yaml` (`kalman: enabled: ...`).

**With the filter:**

```bash
setsid nohup python -u scripts/sb3/train.py --task AS-Tello-Room-Single-v0 --headless --num_envs 512 --max_iterations 1500 \
  env.kalman.enabled=True > outputs/train_room_single_kalman.log 2>&1 &
```

**Without the filter** (the raw per-frame fix, as `core/positional.py` alone gives it):

```bash
setsid nohup python -u scripts/sb3/train.py --task AS-Tello-Room-Single-v0 --headless --num_envs 512 --max_iterations 1500 \
  env.kalman.enabled=False > outputs/train_room_single_raw.log 2>&1 &
```

- **Running both at once** to compare them works (separate logs and run folders), at about
  half speed each. In TensorBoard, tell them apart by start time, or check each run's `params/env.yaml`.
- **Deploying:** a policy trained with the filter must fly with the filter, and one trained
  without it must fly without it. On the laptop, use `FilteredPositioner` from
  `positional_kalman_filter.py` in place of `Positioner`.
- **Filter tuning** (`env.kalman.<field>=<value>`, same values on the laptop: `KalmanCfg` in that file):

  | Field | Default | Meaning |
  |---|---|---|
  | `accel_sigma` | `1.5` m/s² | how hard the drone may accelerate between ticks (higher: follows faster, smooths less) |
  | `meas_sigma` | `0.15` m | noise assumed on one fix (higher: trusts fixes less, smooths more) |
  | `gate_chi2` | `16.27` | outlier gate: fixes further than this (Mahalanobis²) from the prediction are dropped |
  | `max_rejects` | `3` | this many dropped in a row: restart from the fix |

- **Filter self-test, no Isaac needed:** `python3 ../drone_deployment_infra/core/positional_kalman_filter.py`.
  Pass: `SELFTEST PASSED`.

Two training runs at once share the GPU, and each goes about half speed.

### With rendering (watch it learn)

Run these at the desktop: they open the Isaac Sim window. No `setsid nohup ... &`: closing
the window ends the run. The **first** GUI launch compiles shaders for a minute or more
with a blank window: wait, don't Ctrl-C. It learns slower than headless; use it to check
behaviour.

**Every env is overlapped in one room** (the default, `env.scene.env_spacing=0.0`): all the
drones of the batch fly in the same room, so you see the whole batch at once: where the policy
sends drones, how many reach the waypoint, where they land. Overlapping changes only the
picture, not the training: PhysX filters contacts between envs, and crash / collision /
rewards use each env's own positions (measured: no slower than a grid).

**All 512 envs, with the Kalman filter:**

```bash
python -u scripts/sb3/train.py --task AS-Tello-Room-Single-v0 --num_envs 512 --max_iterations 1500 \
  env.kalman.enabled=True env.debug_vis=True > outputs/train_room_single_gui_kalman.log 2>&1
```

**All 512 envs, without the Kalman filter:**

```bash
python -u scripts/sb3/train.py --task AS-Tello-Room-Single-v0 --num_envs 512 --max_iterations 1500 \
  env.kalman.enabled=False env.debug_vis=True > outputs/train_room_single_gui_raw.log 2>&1
```

**Fewer drones on screen** (16 envs, a lighter window; learns much slower):

```bash
python -u scripts/sb3/train.py --task AS-Tello-Room-Single-v0 --num_envs 16 --max_iterations 1500 \
  env.debug_vis=True > outputs/train_room_single_gui.log 2>&1
```

To lay the envs out side by side instead (rooms 14 m apart), add `env.scene.env_spacing=14.0`.

- `env.debug_vis=True` draws the markers (see **In the frame** below). Leave it off headless.
- Camera: `viewer` in `tello_room_env_cfg.py` (env 0, the "overview" view of `record_room.py`).
  Isaac Sim's *Viewer Settings* panel (IsaacLab tab) can move it to another env.
- Any of these also works headless: add `--headless` (and `setsid nohup ... &`) for full speed.

PPO settings are in `tasks/direct/tello_room/agents/sb3_ppo_single_cfg.yaml` (or
`sb3_ppo_cfg.yaml` for 4 drones). Override them the same way: `agent.learning_rate=1e-4`.

---

## 5. Dashboard (TensorBoard from anywhere)

```bash
dashboard/serve.sh     # prints a public https://….trycloudflare.com URL (also in dashboard/URL.txt)
dashboard/stop.sh
```

It shows every run under `logs/sb3/`, and new runs appear on their own. The quick-tunnel URL
is public and changes each time you restart it. See `dashboard/README.md`.


### Watch the simulation live (Isaac Sim WebRTC stream, over Tailscale)

This is the real Isaac Sim viewport, streamed live: the newest checkpoint flying in real time,
16 envs overlapped in one room, with the markers on.

**On the desktop:**

```bash
setsid nohup python -u scripts/stream_room.py > outputs/stream_room.log 2>&1 &
grep "\[stream\]" outputs/stream_room.log      # "streaming ...", then one line per finished episode
```

- **Keeps up with training:** every 5 minutes it swaps in the newest checkpoint (`--reload_min`).
- **Options:**
  - `--checkpoint <zip>`: fly one fixed checkpoint
  - `--num_envs 8`: fewer drones on screen
  - `--task AS-Tello-Room-v0`: the 4-drone task
- **To stop it:** `pkill -TERM -f "[s]tream_room.py"`
- **Not `sb3/play.py --livestream`:** the stream loads an older numpy that can't read the
  checkpoint files, so `play.py` crashes. `stream_room.py` converts the checkpoint first,
  like the recorder.

**On the computer you watch from:**

1. **Install the client:** NVIDIA's *Isaac Sim WebRTC Streaming Client* (Isaac Sim docs, "Livestream
   Clients" page; Linux AppImage, Windows, macOS). Use the version made for Isaac Sim 5.x.
2. **Join the tailnet:** be on the same Tailscale network as the desktop (`tailscale status` lists it).
3. **Connect:** in the client, enter the desktop's Tailscale address, **`100.122.106.39`**, and
   connect. On the home network, `192.168.68.58` works too.

**Limits:**
- **Tailscale only:** the stream doesn't go through the Cloudflare links (they only carry web
  pages), so you need Tailscale or the same local network.
- **Cost:** it uses about 5 GB of GPU memory and some GPU time, so training is a little slower
  while it runs. With the video page also recording, the 16 GB card can run out of memory: run
  one or the other.
- **One viewer:** one client at a time.

### Watch the simulation render from anywhere (video page)

```bash
dashboard/render_serve.sh      # prints a public https://….trycloudflare.com URL (also in dashboard/RENDER_URL.txt)
dashboard/render_stop.sh
```

- **What it shows:** while training runs, `dashboard/render_loop.py` records one full episode
  of the newest checkpoint with `record_room.py`, at most once every 30 minutes. It uses the
  same task and Kalman setting the run trains with.
- **The page:** it plays the newest video and lists the earlier ones (the last 30). When a new
  one is ready it shows a "new video" button, so playback is never cut off.
- **Cost:** each recording takes about 10 min of GPU time while training runs (about 3 min on an
  idle GPU), so training slows a little while it records.
- **Options:** add them after `render_serve.sh`:
  - `--min_gap 900`: record every 15 min
  - `--max_seconds 30`: shorter clips, less GPU time
  - `--view top`: a top-down camera
- **Link:** it has its own URL, separate from TensorBoard. The URL is public and changes each
  time you start the page.
- **Live GUI:** a live stream of the Isaac Sim window (Isaac's WebRTC livestream) doesn't work
  through this kind of tunnel, so the page shows recorded episodes instead.
---

## 6. Watch a trained policy

### Video (headless, mp4)

```bash
python -u scripts/record_room.py --task AS-Tello-Room-Single-v0 --episodes 3 \
  --out outputs/videos/room_single.mp4 > outputs/record_room_single.log 2>&1
grep "\[rec\]" outputs/record_room_single.log
```

- **Checkpoint:** with no `--checkpoint`, it records the newest checkpoint of the newest
  run, so it works mid-training. `--checkpoint <path.zip>` picks one.
- **Real time:** one frame per rc command.
- **Options:**
  - `--view overview|side|top|landing`: `landing` gives a close-up of the target area
  - `--pads 0 1 2 3`: single-drone attempts on each pad in order; overrides `--episodes`
  - `--hide_waypoints`: hide the waypoint spheres when they obscure the drone
  - `--seed N`: different starts and spot assignments
  - `--no_dr`: nominal dynamics
  - `--pilot zero`: no policy, a scene check
  - `--set field=value`: same fields as `env.`
- **In the frame** (and in the GUI with `env.debug_vis=True`), each drone has one color
  (single drone: orange), shared by:
  - small sphere: its true position
  - cube: its published tag fix: where the drone *believes* it is, i.e. the position the
    policy observes. It comes only from AprilTag sightings, 0.2 s late, with 3 cm noise and
    a 3 cm bias, so it trails and jitters around the sphere; it freezes when no tag is in view
  - translucent sphere: its waypoint
  - disc: its spot
  - bottom panel: each drone's phase (`TAKEOFF`, `to waypoint`, `at waypoint`, `LANDING`,
    `LANDED on/OFF spot`, `LOST`), the rc it sends and its fix age.

### Live, with the GUI

```bash
python -u scripts/sb3/play.py --task AS-Tello-Room-Single-v0 --num_envs 64 --real-time env.debug_vis=True
```

All 64 drones fly in the same room (overlapped). Pass the same `env.kalman.enabled=...` the
policy was trained with.

### Score a checkpoint (headless, numbers only)

```bash
python -u scripts/sb3/play.py --task AS-Tello-Room-Single-v0 --headless --num_envs 64 --episodes 256 \
  --checkpoint logs/sb3/AS-Tello-Room-Single-v0/<run>/model.zip > outputs/play_single.log 2>&1
grep "episodes\]" outputs/play_single.log | tail -1
```

The evaluator restores saved config automatically; explicit `env.*` overrides win. It seeds
after loading PPO and collects a fixed quota per simulator slot. `--report outputs/eval.json`
saves individual episode IDs, target assignments, results, and per-pad Wilson intervals.
Use `--config-source current` only to deliberately test a checkpoint under changed conditions.
Legacy runs have no external-asset snapshot, so their historical replay is approximate.

New runs save `params/run.json`, `params/source.tar.gz`, `params/working.diff`, dependency
versions, and copies/hashes of the map, tag sizes and motion fit. Missing normalization is an
error. The config snapshot is taken after motion-fit application.

```bash
python -u scripts/sb3/evaluate_suite.py --checkpoint <run>/model.zip --scenario full --out outputs/full_suite
python -u scripts/sb3/evaluate_suite.py --checkpoint <run>/model.zip --scenario dropout --out outputs/dropout_suite
```

Default suites cover three seeds and all four pads. Promotion requires at least 95% accurate
touchdowns and zero crashes in each seed/pad cohort; inspect confidence intervals too.
The one-drone checkpoint cannot be loaded as a four-drone central controller: their input and
output dimensions differ. Team training remains a separate step after single-drone validation.

---

## 7. What the policy sees and sends

**Observation:** per drone, 24 numbers, plus 1 taking-off flag in the single-drone task.
PPO stacks the last 8 steps. The numbers:

- position (x, y, z): the Kalman estimate (`env.kalman.enabled=False`: the last raw tag fix), and heading
- whether a fix arrived this tick, and seconds since the last fix
- `lost`: no fix for 0.4 s. The position freezes until a tag is back in view (the filter stops
  extrapolating and restarts from the next fix); there is no odometry
- waypoint and landing spot, relative, in the drone's heading frame
- waypoint-reached flag
- bearing to tag 10
- Tello height `h`
- its previous command
- landing flag

Noise on the fixes:
- the camera model: field of view, range, view angle, pixel size, dropout, 0.2 s video delay
- view angle: a tag in the frame decodes on 90% of frames when seen within 10 deg of face-on,
  falling to 0 at 62 deg (`tag_sensor.view_full_angle`, `max_view_angle`)
- motion blur: a whole frame is lost with probability 1.0 per rad/s of yaw rate above 0.2 rad/s
  (at most 0.45) combined with 0.05 per m/s of speed above 0.5 m/s, at most 0.5 in total
  (`tag_sensor.blur_*`). Turning, not straight flight, is what loses the tags. Fitted
  to real flights (view angle: 030-034 and 040-043; blur: 030-034)
- 3 cm noise per fix
- 3 cm bias per drone
- 0–15% link loss per drone

**Kalman filter (on by default):** constant velocity per axis. It predicts every tick, fuses
each fix, and drops outlier fixes (e.g. pose flips). Same file and settings as on the laptop. In
the checks, flying at rc 35, it puts the position ~10 cm from the truth vs ~13 cm for the raw fix.

**Action:** the real Tello commands.
- `rc right forward up yaw`: integers −100…100, sent once per 74 ms step (13.5 Hz, the
  measured flight rate), with 0–16 ms jitter.
- By default, the fifth output above 0.5 for three steps requests `land`.
- Both requested landing and optional `env.auto_land=True` require the same gate: waypoint
  reached, projected XY error <=15 cm, horizontal speed <=0.1 m/s, height within 20 cm of the
  waypoint, position std <=12 cm, velocity std <=0.35 m/s, and accepted-fix age including video
  delay <=0.35 s, continuously for 0.5 s. Lost, uninitialized, or non-finite estimates block it.
  These are initial simulation tuning values, not hardware-validated limits.

`drone_deployment_infra/core/room_policy.py` implements inference, map/camera/body transforms,
normalization-before-frame-stacking, RC rounding, and the shared gate. Its CLI replays recorded
feeds offline and sends no commands. The flight engine owns takeoff, command acknowledgement,
and touchdown detection. Record full filter covariance and accepted-fix timestamps for replay.

**Dynamics:** randomized per drone, per episode. No wind.
- thrust-to-weight ×0.95–1.05
- rotor drag ×0.5–2.0
- body-rate PID: kp/ki ×0.85–1.15, kd ×0.7–1.3
- rc → speed gains from `control/tello_motion_fit.yaml`, fitted from flight 027

**Takeoff (single-drone task):**
- The drone starts on the floor with motors off.
- The firmware climbs at 0.5 m/s to 0.8–1.0 m, ignoring sticks and `land` (about 2.2 s),
  then the policy flies.
- Takeoff steps pay no no-tag loss; time cost and version-2 potential shaping continue.

To start the 4-drone task from the floor too, set `start_on_floor = True` in its config.
That changes its observation size, so it needs a fresh training run.

---

## 8. Files

| Path (under `source/action_space_kit/action_space_kit/`) | What |
|---|---|
| `tasks/direct/tello_room/tello_room_env_cfg.py` | **all knobs**: rewards, geometry, noise, dynamics ranges |
| `tasks/direct/tello_room/tello_room_single_env_cfg.py` | single-drone overrides |
| `tasks/direct/tello_room/tello_room_env.py` | the env: rewards (`_get_rewards`), dones, observation, reset |
| `tasks/direct/tello_room/agents/sb3_ppo*_cfg.yaml` | PPO settings |
| `control/rc_action.py`, `control/tello_motion_fit.yaml` | rc → speed (fitted from real flights) |
| `tags/` | room map, tag sensing model, tag textures |
| `tags/kalman.py` | runs the laptop's filter (`drone_deployment_infra/core/positional_kalman_filter.py`) in the sim, on torch |
| `rl/sb3_shared.py`, `rl/policy_export.py` | SB3 wrappers; checkpoint → TorchScript |

| Script (`scripts/`) | What |
|---|---|
| `check_room_env.py` | environment checks |
| `sb3/train.py`, `sb3/play.py` | train / score / watch (GUI unless `--headless`) |
| `record_room.py` | video |
| `render_room_overview.py` | one annotated still of the room |
| `fit_tello_motion.py` | re-fit the motion from new `recordings/*.train.jsonl` |

The room itself is read live from `drone_deployment_infra/config/map.yaml` + `tags.yaml`.
Tag 10's centre is 1.22 m above the floor (`tag10_height`, measured ~48 in). The left wall,
facing tag 10, is 2 m left of tag 10's centre (`room_y_min` = -2.0).

---

## 9. Troubleshooting

| Symptom | Fix |
|---|---|
| `No module named isaaclab` / wrong python | redo the shell setup (step 1); `CONDA_PREFIX` must be unset |
| log stops with no error | Isaac Sim drops buffered output; you forgot `-u` |
| first launch hangs for minutes / blank GUI window | shader compilation, first run only: wait, don't Ctrl-C |
| GUI opens but no drones visible | add `env.debug_vis=True`; the camera looks at the start box (all envs overlap there) |
| `TELLO_USD_PATH` / asset not found | `export TELLO_USD_PATH=$HOME/isaac/assets/tello.usd` |
| obs shape mismatch loading a checkpoint | the checkpoint is from the other task (25 vs 96 inputs) or an older observation layout |
| training slow | another run or recorder is using the GPU: `nvidia-smi` |

## Research and calibration

The changes retain PPO and the existing Tello RC interface. The [Swift drone-racing
paper](https://www.nature.com/articles/s41586-023-06419-4) supports measuring dynamics and
perception errors on the real platform and incorporating them in simulation. Its thrust/body-rate
actions are not directly available through Tello RC, so its control interface is not copied here.
The [inclined-landing PPO paper](https://arxiv.org/abs/2103.09043) supports starting near a goal
and increasing difficulty gradually. These motivate our experiments; they do not validate our
particular thresholds or establish that this policy transfers to real Tellos.

```bash
python -u scripts/audit_flight_coverage.py '../drone_deployment_infra/recordings/*.train.jsonl' \
  --report outputs/flight_coverage.json > outputs/flight_coverage.log 2>&1
```

The coverage report identifies missing command directions, independent flights, pose rotations
and filter covariance. Fit and hold out entire flights, including different drones. Verify
telemetry velocity units and camera/map/body axes before fitting. The historical fit's one-sided
command envelope (no positive up) is insufficient for this task; do not blindly enable it and
make climbing impossible. No updated dynamics fit should replace the current one until those
data gaps and held-out residuals have been checked.

## Four drones together: retain the learned single-drone policy

### Qualified shared policy (2026-10-14)

The selected `AS-Tello-Room-Peer-v0` policy completed **1,019/1,024 held-out full-room
trials successfully (99.5117%)**. A success requires all four drones to land accurately
on four distinct assigned pads in the same room, without collision, crash or timeout.
There were five collision failures, no crashes, no off-pad trials, and no timeouts.
All 24 assignment permutations were covered. The observed rate's 95% Wilson interval
is 98.86–99.79%; this establishes the reported simulation result, not real-flight reliability.

| Reserved seed | Complete trials | All-four successes | Collisions |
| --- | --- | --- | --- |
| 17291 | 256 | 253 | 3 |
| 41423 | 256 | 256 | 0 |
| 89519 | 256 | 255 | 1 |
| 130073 | 256 | 255 | 1 |

Successful trials averaged 17.68 s. Across their 4,076 drone touchdowns, mean error
was 9.41 cm, the 95th percentile was 16.86 cm, and the maximum was 26.05 cm against
the unchanged 30 cm acceptance radius. The smallest separation in a successful
trial was 27.25 cm; collision detection ran at 500 Hz with a 25 cm threshold.

Use `outputs/qualified_four_drone_policy/model.zip` with its matching
`model_vecnormalize.pkl`, `params/`, and `assets/`. The package pins stage-5 rollout 10
(`model_84760576_steps.zip`), selected after 381/384 development successes and before
any reserved trial ran. It contains the qualification reports, evaluation-source archive,
independent record audit, inference-pipeline log, and preservation audit. Training was
stopped after selection; the original solo checkpoint and its normalizer remain unchanged.

This is one policy shared by four drones: each receives 48 features across eight
frames (384 inputs) and emits five actions every 74 ms (13.51 Hz). A frozen 256×256
original pilot is combined with a learned 128×128 correction and bounded Gaussian
means. Peer inputs use the saved version-2 ordering. The saved landing/takeoff controller,
room geometry, noisy delayed sensing, domain randomization, and 75 s limit are retained.

```bash
python -u scripts/sb3/play.py --headless --task AS-Tello-Room-Peer-v0 \
  --checkpoint outputs/qualified_four_drone_policy/model.zip \
  --num_envs 128 --episodes 128 --seed 123 --report outputs/qualified_replay.json \
  env.collision_substeps=True > outputs/qualified_replay.log 2>&1
```

The four reserved seeds above have now been consumed for this qualification. Further
policy tuning requires a fresh untouched qualification set. The following sections
document the transfers and development experiments that led to the selected policy.

The packaged `demo.mp4` records four drones together in one physics room on seed 7.
It completed four accurate landings in 17.464 s, with 0.5035 m minimum separation
and 6.31 cm mean touchdown error. The departure, approach, landing and terminal
result frames were visually checked, including all four final HUD landing statuses.

### Initial centralized transfer

`AS-Tello-Room-Team-v0` uses the same room, 6–7 m start range, takeoff,
25-feature per-drone observation, and landing gate, with four drones in each room.
The central wrapper concatenates their observations and emits all four action sets:
100 features per frame, 8 frames (800 inputs), and 20 outputs. Each simulated drone
still receives a decision every 74 ms. This is centralized PPO, not MAPPO.

Expand a selected full-mission checkpoint before training; changing `num_drones` on
the original checkpoint cannot change its network dimensions safely:

```bash
python -u scripts/sb3/expand_room_policy.py \
  --checkpoint outputs/full_mission_candidate/model.zip \
  --out outputs/team_from_full_mission > outputs/team_transfer.log 2>&1
python -u -m unittest discover -s tests -p 'test_*.py' > outputs/team_cpu_tests.log 2>&1
python -u scripts/check_room_env.py --headless --task AS-Tello-Room-Team-v0 \
  --num_envs 128 > outputs/check_room_team.log 2>&1
```

Require `TEAM TRANSFER PASSED`, successful CPU tests, and `ALL CHECKS PASSED`.
The transfer preserves the parent's weights and input normalization in four network
blocks. Cross-drone weights initially equal zero but remain trainable. A 256×256
parent becomes a 1024×1024 actor and critic. Frame ordering, raw actor outputs,
critic values, normalization, reset padding, and serialization are checked.
The normalized joint reward uses mean per-drone units; the initial critic predicts
the mean of the parent's four value estimates. The original artifact is unchanged.

The expanded artifact is an **untrained coordination initialization**, not a successful
four-drone policy. Measure it with `sb3/play.py --task AS-Tello-Room-Team-v0`, then
fine-tune using `sb3/train.py --task AS-Tello-Room-Team-v0 --checkpoint ...`.
Keep the matching normalizer with each checkpoint. Use frozen normalization during
initial fine-tuning so the transferred navigation behavior retains its input scale.
The joint critic initially estimates independent returns; fit it before updating the
flight actor, then use small PPO updates. For example:

```bash
python -u scripts/sb3/train.py --headless --task AS-Tello-Room-Team-v0 \
  --checkpoint outputs/team_from_full_mission/model.zip --num_envs 512 \
  --max_iterations 30 --save_interval 10 --seed 52 --action-std 0.10 \
  --critic-warmup-iterations 4 agent.learning_rate=3e-6 \
  agent.n_minibatches=16 agent.target_kl=0.01 > outputs/train_team_conservative.log 2>&1
python -u scripts/check_policy_pipeline.py --headless \
  --checkpoint outputs/team_from_full_mission/model.zip > outputs/check_team_policy_pipeline.log 2>&1
```

The pipeline check compares observations, actions and RC commands through forced
resets, then isolates the landing gates with controlled hover-over-pad fixtures.
It does not establish full-room collision avoidance; use complete flight evaluations
for that. Training success rates also cannot replace a fixed evaluation cohort.

Report joint success (all four accurate landings without violations), collision/crash
rates, timeouts, minimum separation, per-pad touchdown errors, and completion time
on held-out seeds. Record an actual shared physics room with `record_room.py --task
AS-Tello-Room-Team-v0`; `--independent_drones` is only for the original single-policy
baseline. Do not interpret four individually successful flights as a joint success.

### Peer-motion inputs and predictive separation training

`AS-Tello-Room-Peer-v0` retains the original 25 values and appends 23 estimated-state
features per drone: heading-frame velocity (3), maximum position/velocity standard
deviation (2), and the other three drones' relative positions and velocities (18).
Neighbor slots follow ascending drone index, excluding self. The central policy has
48 × 4 × 8 = 1,536 inputs and the same 20 action outputs. Both actor and critic retain
the trained 1024×1024 hidden layers. `core/team_features.py` supplies the identical
NumPy/Torch feature calculation to the offline adapter and simulator; no true velocity
or position is added to the actor input.

```bash
python -u scripts/sb3/extend_team_observations.py \
  --checkpoint logs/sb3/AS-Tello-Room-Team-v0/<selected-run>/model.zip \
  --out outputs/peer_from_team30 > outputs/peer_transfer.log 2>&1
python -u scripts/check_room_env.py --headless --task AS-Tello-Room-Peer-v0 \
  --num_envs 128 > outputs/check_peer_room.log 2>&1
python -u scripts/check_policy_pipeline.py --headless \
  --checkpoint outputs/peer_from_team30/model.zip > outputs/check_peer_pipeline.log 2>&1
```

Require `PEER TRANSFER PASSED`, `ALL CHECKS PASSED`, and `POLICY PIPELINE PASSED`.
The appended first-layer columns start at zero. Existing input statistics, outputs,
critic predictions, and reset padding are preserved; new metre/m/s/std inputs use
fixed unit scaling. The artifact includes the parent configuration and current
source snapshot. Transfer alone does not teach collision avoidance.

The peer task adds an optional reward cost for predicted close approaches. From
relative position and velocity, calculate the closest approach within 1.5 seconds.
The cost grows below a predicted 0.5 m separation and decays with time to that
approach; its default weight is -100 per second. It uses true simulator motion only
for the training reward. This is a learning signal, not a collision-avoidance override
or safety guarantee. Existing room tasks default to zero weight, preserving their
reward behavior. Collision termination remains at 0.25 m.

Relative neighbor position and velocity are also used in
[Huang et al., ICRA 2024](https://arxiv.org/html/2309.13285v2). Their results motivate
providing those features explicitly; they do not validate this Tello implementation.
Our predictive reward and its numerical settings are experiment choices.

For diagnosis, `sb3/play.py ... env.log_terminal_state=True` adds terminal true and
estimated motion, estimator variances, and takeoff/landing state to evaluation records.
This flag affects logging only. Inspect full
cohorts, including collisions, crashes, off-pad landings and timeouts; reserve fresh
seeds for final qualification. A higher training reward is not sufficient evidence
of improved flight reliability.

### Shared peer-aware policy

The same peer task can train one network shared by all four drones. Each drone gets
its own 48-feature observation and eight-frame history (384 inputs), then produces
five actions. All four act simultaneously in the same physics room. The shared
wrapper supplies per-drone rewards and experiences to PPO; there is no centralized
critic. This alternative retains the original single-drone 256×256 actor and critic.

```bash
python -u scripts/sb3/extend_team_observations.py --shared \
  --checkpoint outputs/full_mission_candidate/model.zip \
  --out outputs/shared_peer_from_full_mission > outputs/shared_peer_transfer.log 2>&1
python -u scripts/check_policy_pipeline.py --headless \
  --checkpoint outputs/shared_peer_from_full_mission/model.zip > outputs/check_shared_peer_pipeline.log 2>&1
python -u scripts/sb3/train.py --headless --task AS-Tello-Room-Peer-v0 \
  --checkpoint outputs/shared_peer_from_full_mission/model.zip --num_envs 512 \
  --max_iterations 80 --save_interval 10 --seed 74 --action-std 0.10 \
  --critic-warmup-iterations 4 agent.learning_rate=1e-5 \
  agent.n_minibatches=32 agent.target_kl=0.01 agent.ent_coef=0.0 \
  > outputs/train_shared_peer_stage1.log 2>&1
```

This initialization sets `agent.central=False`, `peer_sort_by_distance=True`, and
`collision_substeps=True`. The three neighbor slots are ordered nearest first from
estimated positions, so drone identifiers do not define their meaning. New input
weights start at zero and the original policy outputs are checked before training.
New transfers save `peer_order_version=2`: distances are ordered using world-axis
positions rounded to integer millimeters, with drone index breaking ties. Actual
observation values remain unrounded. This prevents symmetric starts from producing
different neighbor slots in NumPy/CPU and torch/CUDA because of trigonometric
roundoff. Saved runs without this field retain version 1 for historical replay.
The optimizer and coordination transition count start fresh; the learned weights,
normalization, full-room starts, sensing, dynamics and landing criteria are retained.
With 512 rooms, each rollout contains 512 × 4 × 256 = 524,288 per-drone transitions.
Compare physical room trials rather than raw transition counts between architectures.

`sb3/play.py`, `record_room.py`, and the offline `RoomPolicy` adapter select shared
or central inference from the artifact's saved configuration. A shared peer policy
still requires estimated telemetry for every drone; sharing the network does not
remove that observation requirement. It has no scripted yielding or action filter.

### Learn corrections in command space

The unsquashed Gaussian pilot can produce means well outside the command limits.
A forward mean near 2 with standard deviation 0.1 produces nearly identical clipped
maximum-forward samples. This restricts exploration of braking, even with peer inputs.
Use `sb3/play.py --action-diagnostics --steps 120 ...` for a short probe of raw means
during flying states. A capped probe keeps `complete=false` if its episode cohort is
unfinished and must never be reported as a reliability score.

`initialize_residual_policy.py` adds a 128×128 learned correction network to the
shared peer pilot. Its mean is `clip(base_mean, -1, 1) + correction`; Gaussian noise
and the existing final action limits follow. The base 256×256 flight actor stays
frozen, the correction head starts at zero, and the critic remains trainable. Initial
deterministic commands match the parent exactly, while stochastic exploration can
now produce slower commands near a saturated base output. This is an experimental
learned policy architecture, with no scripted yielding rule or collision override.

```bash
python -u scripts/sb3/initialize_residual_policy.py \
  --checkpoint outputs/shared_peer_from_full_mission/model.zip \
  --out outputs/residual_from_full_mission > outputs/residual_transfer.log 2>&1
python -u scripts/check_policy_pipeline.py --headless \
  --checkpoint outputs/residual_from_full_mission/model.zip > outputs/check_residual_pipeline.log 2>&1
python -u scripts/sb3/train.py --headless --task AS-Tello-Room-Peer-v0 \
  --checkpoint outputs/residual_from_full_mission/model.zip --num_envs 512 \
  --max_iterations 80 --save_interval 10 --seed 85 --action-std 0.10 \
  --critic-warmup-iterations 4 agent.learning_rate=1e-4 \
  agent.n_minibatches=32 agent.target_kl=0.01 agent.ent_coef=0.0 \
  > outputs/train_residual_stage1.log 2>&1
```

Require `RESIDUAL TRANSFER PASSED`, the residual CPU tests, and
`POLICY PIPELINE PASSED` before training. Tests cover preservation of commands,
nonzero braking gradients, frozen base weights, critic warmup, isolated checkpoint
reload/resave, and TorchScript export. The small custom policy classes are embedded
in the checkpoint so CPU inference does not need Isaac Lab imports. The matching
normalizer and saved observation configuration are still required.

This uses the general idea of learning an additive correction described in
[Residual Policy Learning](https://arxiv.org/abs/1812.06298); our base is a learned
flight policy, and the command-clipping choice addresses this project's measured
saturation. That research does not establish this implementation's reliability.

Inspect every failure type before continuing. In the first residual experiment,
side-boundary crashes increased while their individual penalty was much smaller
than a two-drone collision's combined penalty. The next experiment resumes that
checkpoint with `env.rew_violation=-2000.0` and two critic-warmup iterations. This
changes the training incentive; the room boundaries and evaluation criteria stay
fixed. It is not evidence that the revised policy succeeds until evaluation passes.

Completed stage-2 development evaluations (2026-10-14), using deterministic actions,
randomized dynamics and sensing, full-room floor starts, and 500 Hz collision checks:

| Checkpoint rollout | Development seed | Complete trials | All-four successes | Collisions | Crashes |
| --- | --- | --- | --- | --- | --- |
| 10 | 123 | 128 | 58 (45.3%) | 68 | 2 |
| 20 | 123 | 128 | 92 (71.9%) | 35 | 1 |
| 30 | 456 | 256 | 219 (85.5%) | 36 | 1 |
| 40 | 123 | 128 | 119 (93.0%) | 8 | 1 |
| 60 | 456 | 256 | 242 (94.5%) | 13 | 1 |
| Final after 80 | 123 | 128 | 123 (96.1%) | 4 | 1 |
| Final after 80 | 456 | 256 | 249 (97.3%) | 5 | 2 |

There were no timeouts in these cohorts. At rollout 30, one collision also registered
a crash, and one other trial landed off its pad; failure categories can overlap.
Matched-seed comparisons verify identical starts and pad assignments. Rollout 60
fixed 23 failures from rollout 30 without losing a previous success. All 13 remaining
collisions occurred within the first eight seconds. These were development results;
at that stage, the reserved qualification had not run and the 99% target had not been met.

The final stage-2 model completed 372/384 development trials successfully (96.875%):
nine collisions and three side-boundary crashes, all within the first six seconds.
It had no off-pad landings or timeouts. Stage 3 resumes this model with doubled
collision (`-2000`), crash (`-4000`), and predictive-proximity (`-200` per second) costs,
with two critic-warmup iterations. The physical challenge is unchanged. This next
reward experiment is not qualified until its own complete evaluations pass.

Stage-3 rollout 40 completed 126/128 trials on development seed 123 (98.4375%),
with two early collisions and no crashes, missed landings or timeouts. It fixed
three stage-2 failures without losing a previous success on those same trials.
A copy migrated to peer-order version 2 passed the simulator/adapter pipeline
check and repeated the 126/128 result. These scores use the same development
cohort and must not be combined as independent trials.
The migrated copy also scored 252/256 on development seed 456: together it completed
378/384 trials (98.4375%), with five collisions and one boundary crash. Successful
flights averaged 16.51 s and 8.66 cm touchdown error. This is the best completed
two-cohort development result at this checkpoint; it still falls short of 99%.

Reports use `outputs/residual_stage2_<rollout|final>_strict_seed<seed>.json`. The best
stage-2 checkpoint is `model.zip` in `2026-10-14_20-51-13`, with its matching
`model_vecnormalize.pkl`. Rollout-60's frozen pilot and normalization statistics
were checked against the original initialization. A separate seed-7 recording of
rollout 40 completed four accurate landings in 22.274 s, with 0.385 m minimum separation.
Use the `overview` camera; the `side` camera is obstructed by the room's left wall.

New training processes handle SIGINT/SIGTERM by setting a stop request, then return
from PPO at a callback boundary to save the model and normalization. Raising an
exception directly in the signal handler can be swallowed inside an Isaac physics
callback. Require the final save messages and reloadable artifacts after stopping.

### Controlled residual-policy migrations

`prepare_residual_candidate.py` creates a separate candidate while retaining every
parameter tensor, the frozen base pilot, critic and matching normalization. It accepts
final or numbered checkpoints. `--stable-peer-order` saves ordering version 2;
`--bound-mean` additionally changes the Gaussian mean to
`tanh(clip(base_mean, -1, 1) + correction)`. The Gaussian noise and final action limits
are unchanged. Bounded means are an experiment intended to preserve braking
exploration after corrections have grown. They change deterministic commands;
an activation change alone does not establish a reliability improvement.

```bash
python -u scripts/sb3/prepare_residual_candidate.py \
  --checkpoint <parent-run>/model.zip --out outputs/stable_candidate \
  --stable-peer-order > outputs/stable_candidate_transfer.log 2>&1
python -u scripts/check_policy_pipeline.py --headless \
  --checkpoint outputs/stable_candidate/model.zip > outputs/stable_candidate_pipeline.log 2>&1
```

Require `RESIDUAL CANDIDATE TRANSFER PASSED` and `POLICY PIPELINE PASSED`, then
evaluate a complete cohort before training or selecting the candidate. Compare
ordering-only and bounded-mean candidates from the same checkpoint on the same
development trials so their effects can be distinguished. Keep the parent's
checkpoint, configuration and scores separate from either migration.
The initial bounded-mean copy of stage-3 rollout 40 scored 125/128 on seed 123,
with three collisions versus two for the ordering-only copy. It was not promoted;
changing the activation alone did not improve this cohort.

The ordering-only rollout-60 copy later scored 376/384 across the same two cohorts,
so rollout 40 remained the reference for that stage. Stage 3 was stopped gracefully
and its final state saved separately for evaluation. Stage 4 trained the bounded-mean rollout-40
copy with 512 rooms, seed 118, learning rate `5e-5`, initial action standard deviation
`0.10`, and two critic-warmup iterations. The stage-3 avoidance costs are retained.
This tests learning with bounded means; the 125/128 pretraining score does not
establish that the new training run improves reliability.

The first stage-4 checkpoint completed 380/384 development trials (98.9583%):
125/128 on seed 123 and 255/256 on seed 456, with four collisions and no crashes,
missed landings or timeouts. Successful flights averaged 18.22 s and 9.02 cm touchdown
error. This improves the combined development score but must not be rounded up to
a 99% pass. Its pipeline check passed, and an artifact audit verified the frozen
pilot, unchanged observation/return normalization, original checkpoint hash, and
unchanged physical and landing challenge. The artifact is
`2026-10-14_22-40-40/model_74274816_steps.zip` with its matching numbered normalizer.

All four remaining collisions occurred within 4.15 s after reset. Terminal diagnostics
show all drones had exited takeoff mode. In two of these collisions, the estimated
pair separation was approximately 0.50–0.59 m while true separation was 0.23 m.
The simulation still includes the original video delay of three control steps and Kalman filter;
the policy must anticipate motion rather than rely on delayed separation alone.

Stage-4 rollout 20 repeated 380/384 with the same four collision cases. The run was
stopped gracefully, saving its final state separately, and stage 5 resumed the
evaluated rollout-20 checkpoint. It raised collision cost from `-2000` to `-8000`
and crash cost from `-4000` to `-16000`, retaining predictive cost `-200` per second,
learning rate `5e-5`, 512 rooms, and all physical and landing criteria. Four
critic-warmup iterations let the value network adapt before actor updates. This
tested whether a greater cost for rare failures encourages more cautious departures.
Its rollout-10 checkpoint improved development success to 381/384 (99.21875%),
with three collisions and no crashes, off-pad landings or timeouts. Selection was
then frozen for the successful reserved qualification summarized above.

### Final joint reliability qualification

New environments check the 0.25 m separation limit at every 2 ms physics step with
`collision_substeps=True`. Violations are retained until the next policy update,
including a pair that separates again before that update. Older saved configurations
retain command-step checking when replayed; explicitly pass
`env.collision_substeps=True` when comparing them against new candidates. This changes
collision measurement, not the flight controller or physical dynamics.

Only after development evaluations meet the target, run the reserved qualification:

```bash
python -u scripts/sb3/qualify_room_policy.py --checkpoint <selected-run>/model.zip \
  --out outputs/team_qualification > outputs/team_qualification.log 2>&1
```

This runs 256 full-room trials on each of four reserved seeds (1,024 total), with
physics-substep collision checks. It verifies the challenge against the preserved
`outputs/full_mission_candidate/params/run.json`, checks room/sensor/dynamics assets,
and audits individual records rather than accepting a summary alone. It also retains
the original landing gates, controller settings and policy decimation. Qualification
requires at least 99% all-four successes, all 24 drone-to-pad assignments represented,
full 6–7 m floor starts, and the original 75 s / 0.3 m touchdown criteria. Successful
trials must have no collisions, crashes or timeouts. The report includes a confidence
interval; an observed 99% simulation score is not a guarantee of 99% real-flight
reliability. The evaluation source is archived beside the results.
