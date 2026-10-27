# Action Space kit: setup and run guide (step by step, with pass criteria)

> **Policy reliability updates (2026-10-14):** new room runs use `reward_version=2` and
> `land_gate_enabled=True`, with policy-requested landing (`auto_land=False`). The gate
> checks current estimated position, velocity, covariance, accepted-fix age and 0.5 s dwell.
> New waypoint height is 1.22 m (measured tag center), improving tag visibility over the pads.
> The older reward/auto-land descriptions below describe legacy checkpoints; use the updated
> `simulation_infra/SINGLE_DRONE_SIMULATION.md` for current defaults and curriculum commands.
> `sb3/play.py` restores saved config, seeds after PPO loading, and counts a fixed episode
> quota per env. Save `--report outputs/<name>.json`; require `complete=true` and exact counts.
> Resume requires matching VecNormalize stats. New runs snapshot assets, effective config,
> source, and dependencies. A single-drone checkpoint cannot resume as a four-drone model.
> Before long training, run `tests/test_policy_contract.py` and the single-drone environment
> checks. Use `check_policy_pipeline.py` with a short two-second checkpoint to check inference
> parity and all four controlled landing targets. No hardware commands are sent by these checks.

> **Repo layout (moved 2026-10-07):** the Isaac Lab **simulation** now lives in
> `simulation_infra/` — prefix the paths in the steps below (`setup.sh`, `source/`,
> `scripts/`, `policies/`, `assets/`, `policy_maker.md`) with `simulation_infra/`
> (e.g. `simulation_infra/setup.sh ~/isaac`). The **real-world drone** code is in
> `drone_deployment_infra/` (calibration, mapping, connect, policies, recordings) —
> see `drone_deployment_infra/README.md`. This guide covers the simulation side.

Isaac Lab kit for the Action Space hackathon. The main demo is **`AS-Tello-Approach-v0`**:
four DJI Tello drones fly to a base, each passes through its own gate on a 3 m keep-out
sphere, then lands on its own pad. The overview version of this guide is the
"Installation" section of `ACTION_SPACE_KIT.md`.

Last verified end-to-end: 2026-09-27, RTX 5080 16 GB, driver 580.95.05, Linux Mint 22.3
(Ubuntu 24.04 base), Isaac Sim 5.1.0, Isaac Lab v2.3.2, torch 2.7.0+cu128, Python 3.11.

## Ground rules for agents

- **Run every step and check its pass criterion before moving on.** Don't report success
  from exit codes alone. Isaac Sim often exits 0 after a failure, and it hard-exits on
  `simulation_app.close()`, so buffered stdout gets lost. **Always run Python with `-u`**
  and redirect to a log inside the repo (`outputs/` is gitignored), for example
  `python -u ... > outputs/<name>.log 2>&1`, then grep the log.
- Don't write logs, frames or videos to `/tmp` or `/mnt/data`. Use `outputs/` in the repo.
- `setup.sh` may call `sudo apt-get`, but only for missing packages. If sudo needs a
  password you don't have, check `dpkg -s cmake build-essential git git-lfs curl ffmpeg
  libglu1-mesa libvulkan1 vulkan-tools`. If anything is missing, ask the human to run
  `! sudo apt-get install -y <pkgs>`.
- Filter logs when reading them. Isaac Sim prints hundreds of harmless `[Warning]` lines.
  Useful filter: `grep -vE "Warning\]|WARNING|^\s*$|^\|"`.

## Step 0: preflight

```bash
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
df -h ~          # need ~30 GB free
echo "CONDA_PREFIX=$CONDA_PREFIX"   # non-empty is fine: setup.sh unsets it, but you must too (Step 2)
```

Pass: a GPU is listed. RTX 50-series needs driver >= 570. If `nvidia-smi` fails, stop and
tell the human: it's a driver problem (often a kernel upgrade left the module unbuilt).

## Step 1: install

```bash
git clone https://github.com/jaromero3rd/Action_Space.git ~/Action_Space   # skip if already cloned
cd ~/Action_Space
mkdir -p ~/isaac
./setup.sh ~/isaac > ~/isaac/setup.log 2>&1; echo "EXIT=$?" >> ~/isaac/setup.log
grep -E "^== |EXIT=|Traceback|wrote|isaaclab ok|AS-Tello-Approach" ~/isaac/setup.log
```

Takes 7-40 minutes depending on bandwidth (7 on the reference box). Run it in the
background and watch the `== ` stage markers. Stages, in order: `checking GPU`,
`system packages`, `workspace`, `python 3.11 environment`, `Isaac Sim 5.1.0`,
`PyTorch (CUDA 12.8)`, `Isaac Lab v2.3.2`, `this kit`, `Tello asset`, `verifying`.

Pass, all of:
- `EXIT=0`
- `wrote <ISAAC_ROOT>/assets/tello.usd` (or the file already existed)
- `isaaclab ok, cuda: True`
- a table row containing `AS-Tello-Approach-v0`

The script is idempotent; re-run it after fixing a failure.

What it produces:
| Path | What |
|---|---|
| `~/isaac/env_isaaclab/` | Python 3.11 uv venv with isaacsim, isaaclab, torch, skrl, stable_baselines3, this kit (editable) |
| `~/isaac/IsaacLab/` | Isaac Lab v2.3.2 checkout |
| `~/isaac/assets/tello.usd` | drone model, converted from `assets/tello/tello.urdf` |
| `~/isaac/.cache`, `~/isaac/tmp` | pip/uv caches and TMPDIR used during install |

Data the demo needs that's **already in git** (verify with `file`: they must be
`Zip archive` / `data`, not ASCII LFS pointers):
`policies/approach_it5_room_fixed/model.zip`, `model_vecnormalize.pkl`, `params/*.yaml`,
`command.txt`.

### Known install traps (all handled by the current setup.sh)

| Symptom in the log | Cause | Handling |
|---|---|---|
| `ERROR: pip's dependency resolver ... packaging 23.0` | Isaac Sim pins packaging 23.0 | harmless, ignore |
| `FileNotFoundError: Could not find the isaac-sim directory: .../IsaacLab/_isaac_sim` right after `Setting up vscode settings` | an active conda env: `isaaclab.sh` prefers `$CONDA_PREFIX` python over `$VIRTUAL_ENV` | setup.sh unsets the `CONDA_*` vars |
| `error: Distribution not found at: file://<ISAAC_ROOT>/source/action_space_kit` | old setup.sh resolved `$0` after `cd` | fixed: `KIT_DIR` is resolved at the top |
| `No module named isaaclab` after install | flatdict build failure made Isaac Lab skip the core package | setup.sh pins `setuptools<81`, prebuilds flatdict, then reinstalls `source/isaaclab` |
| `[WARN] Could not find Isaac Sim VSCode settings` | cosmetic | ignore |

## Step 2: shell environment (every new shell / every Bash call)

Shell state doesn't persist between agent tool calls, so prefix each command:

```bash
cd ~/Action_Space && unset CONDA_PREFIX CONDA_DEFAULT_ENV && source ~/isaac/env_isaaclab/bin/activate && export OMNI_KIT_ACCEPT_EULA=YES TELLO_USD_PATH=$HOME/isaac/assets/tello.usd && mkdir -p outputs
```

`TELLO_USD_PATH` is mandatory. Its default (`/mnt/data/isaac/assets/tello.usd`) exists only
on the old shared server.

## Step 3: verify the environment (~2 min)

```bash
python -u scripts/check_approach_env.py --headless --test all > outputs/check_approach.log 2>&1
grep -E "^\s*\[(ok|FAIL)|^== |zero |scripted " outputs/check_approach.log
```

Pass: every check line is `[ok]` (12 geometry, 6 controller, 4 baseline) and the baseline
lines read roughly:
```
zero     : success 0%,   ... team return ~ -800
scripted : success 100%, arrived 4.00/4, landed 4.00/4, gates 4.00/4, illegal entry 0%, collision 0%, ... team return ~ +1540
```
The first ever launch compiles shaders for several minutes; that's not a hang.

## Step 4: score the shipped trained policy (~1 min)

```bash
python -u scripts/sb3/play.py --task AS-Tello-Approach-v0 --headless --num_envs 64 --episodes 256 \
  --checkpoint policies/approach_it5_room_fixed/model.zip \
  env.spawn_azimuth_range=0.0 env.rew_scale_separation=-20.0 env.rew_violation=-100.0 \
  > outputs/play_policy.log 2>&1
grep "episodes\]" outputs/play_policy.log | tail -1
```

Pass (reference run): `near_base_frac=0.897, success_rate=0.864, collision_rate=0.136,
gates_passed=4.000, illegal_entry_rate=0.000, crash_rate=0.000`. Treat `near_base_frac >= 0.80`
as the bar. The three `env.*` overrides are **required**; they match the training
command in `policies/approach_it5_room_fixed/command.txt`. `play.py` loads
`model_vecnormalize.pkl` from next to the checkpoint automatically.

## Step 5: record a video (~1-2 min)

```bash
python -u scripts/record_approach.py --pilot policy \
  --checkpoint policies/approach_it5_room_fixed/model.zip \
  --spawn_azimuth_range 0.0 --set rew_scale_separation=-20.0 rew_violation=-100.0 \
  --episodes 2 --out outputs/videos/approach_policy.mp4 --frame_dir outputs/frames \
  > outputs/record.log 2>&1
grep "\[rec\]" outputs/record.log
```

Pass: `[rec] wrote outputs/videos/approach_policy.mp4 (<~1 MB+> bytes)` and at least one
`episode N: SUCCESS -- gates 4/4, landed 4/4`. An episode may show `cut at 20 s`, which
is the recorder's cap, not an error. **Always pass `--out` and `--frame_dir`**; the
defaults are under `/mnt/data`. To confirm frames aren't blank, extract one and look at it:
`ffmpeg -loglevel error -y -ss 7 -i outputs/videos/approach_policy.mp4 -frames:v 1 outputs/frame.png`.
Expect a grid floor, a red dotted sphere, four colored gate rings and a green touchdown disc.
Delete `outputs/frames` afterwards.

Viewport capture works headless because the recorder enables livestream mode internally.
Replicator camera *sensors* are a separate known issue (see README), and this demo doesn't use them.

## Where things live

| Path | What |
|---|---|
| `source/action_space_kit/action_space_kit/tasks/direct/tello_approach/` | approach task: `tello_approach_env_cfg.py` (geometry, rewards: every `env.*` override), `tello_approach_env.py`, `scripted_pilot.py` |
| `source/action_space_kit/action_space_kit/control/` | Tello firmware/velocity cascade model (`tello_dynamics.py`), rc-channel controller |
| `source/action_space_kit/action_space_kit/rl/sb3_shared.py` | makes all 4 drones share one SB3 policy |
| `source/action_space_kit/action_space_kit/assets/tello.py` | reads `TELLO_USD_PATH` |
| `scripts/sb3/{train,play}.py` | SB3 train / score |
| `scripts/check_approach_env.py`, `scripts/record_approach.py` | verification, video |
| `policies/` | shipped checkpoints (small, tracked in git despite the `.gitignore` defaults) |
| `setup.sh` | the installer; `assets/tello/convert.sh` is URDF→USD |

## Training (optional)

For reward locations, tuning knobs, batch sizing and measured throughput, see `policy_maker.md`.

```bash
python -u scripts/sb3/train.py --task AS-Tello-Approach-v0 --headless --num_envs 512 --max_iterations 800 \
  agent.learning_rate=1e-3 env.spawn_azimuth_range=0.0 env.rew_scale_separation=-20.0 env.rew_violation=-100.0
```

Output goes to `logs/sb3/AS-Tello-Approach-v0/<timestamp>/model.zip`. For runs longer than
your tool timeout, launch detached (`setsid nohup ... > outputs/train.log 2>&1 &`) and
poll the log. Score the result with the Step 4 command, pointing `--checkpoint` at it.

## Room task: `AS-Tello-Room-v0` (real map, AprilTag-only sensing, real rc)

Human guide for both room tasks (train, tune rewards, watch): `simulation_infra/SINGLE_DRONE_SIMULATION.md`.

Four Tellos start hovering in a 2 m x 2 m box 6-7 m from tag 10 (>= 0.2 m apart, `spawn_min_separation`;
a pair spawned closer than `collision_distance` collides only below its start distance - 3 cm), fly to four
landing spots on the floor 1 m in front of the tag-10 wall, 0.5 m apart, and send `land` over
them. Start positions and the drone-to-spot assignment are random every episode. Each drone faces tag 10
+-15 deg, or (`spawn_search_prob` 0.3) starts turned 75-180 deg away, with no tag in view, and must turn to find one. Success: all
four landed within 0.3 m of their spots.
Trained **end to end** by **one central controller** (the laptop): a single PPO policy sees all
four drones' positional feeds and outputs all four drones' commands (agent cfg `central: true`,
`rl/sb3_shared.Sb3CentralVecEnv`), trained on the team reward. No scripted pilot, search or hold.
Run from `simulation_infra/` with the Step 2 environment.

- **Observation** (96 = 4 drones x 24, PPO stacks 8): per drone, its `core/positional.py` feed as
  the laptop receives it: last tag fix, heading, fix age, `lost` (no fix for 0.4 s: its position
  freezes until it turns back to a tag; no odometry), its waypoint and landing spot in its heading
  frame, waypoint-reached flag, tag-10 bearing, Tello `h`, its previous command, landing flag.
  Fixes carry camera-model noise, 3 cm per-fix noise and a 3 cm per-drone bias; frames are lost
  on the link at a per-drone, per-episode rate.
- **Waypoint**: 1.5 m above each landing spot, radius 0.3 m. Entering it counts only at <= 0.3 m/s
  (`waypoint_max_speed`, Kalman speed estimate, judged once its velocity std < `speed_trust_std` 0.35);
  faster ends the episode with -500 per 0.3 m/s over (`fast_entry_loss`). Within 2 m of tag 10
  (`slow_zone_radius`) `slowdown_reward` pays +200 per m/s of new low true speed (flights 032-034 crossed it at ~2 m/s). Until a drone's published fix is
  inside it, `closer_reward` pays +100 per metre gained toward the waypoint (3D); after that only
  landing rewards (touchdown, formation) and the penalties apply.
- **Per-drone dynamics**: TWR x0.95-1.05, rotor drag x0.5-2.0, body-rate PID kp/ki x0.85-1.15,
  kd x0.7-1.3, plus outer-loop gains, motor lag, mass, latency, rc->speed gain. No wind.
- **Actions**: the real Tello commands: the 4 rc channels, each an integer in -100..100 (the full
  SDK range), plus `land` (5th output > 0.5 for 3 steps). The logs' sent range is still recorded
  as `command_envelope` in the fit file (`rc.use_envelope=True` would restrict to it); speeds past
  the logged sticks (~21-35) extrapolate the fitted gains.
- **Rewards** (per drone-step, summed over the drones for the controller): +100 per metre closer
  to the waypoint until reached, -25 per metre away (`closer_reward`, `rew_away_ratio` 0.25),
  +1000 once on entering the waypoint (`waypoint_reward`, `rew_waypoint`), -0.5 per step
  until landed on the spot (`landing_loss`; ending early pays the rest), -0.5 flying without a tag,
  -100 crash / wall (`crash_loss`), -1000 per drone within 0.25 m of another (`collision_loss`, `rew_collision`), +100 at touchdown (`land_reward`), +200 all landed on spot.
- **Auto land** (`env.auto_land`, on by default): the step after a drone's published fix enters
  its waypoint (the obs waypoint-reached flag) `land` is sent automatically and the policy's 5th
  output is ignored. The laptop must do the same at deploy. With `env.auto_land=False` the policy
  sends land itself, gated: before the waypoint it is blocked and costs -100 (`early_land_loss`).
  Each reward term is logged as
  `Rewards/avg_<term>_per_episode` (per drone).
- **Overlapped envs**: `scene.env_spacing` defaults to 0, so every env shares one room (the GUI
  shows the whole batch). PhysX filters contacts between envs and every reward/crash uses the
  env's own positions; measured no slower than the 14 m grid (`env.scene.env_spacing=14.0`).
- **Room**: read live from `drone_deployment_infra/config/map.yaml` + `tags.yaml` (no copy;
  re-mapping the room updates the sim). Tag 10's wall is X = 0, Z up from the floor; the floor
  isn't in the map, so `env.tag10_height` (1.22 m, measured ~48 in) places it. The left wall
  (facing tag 10) is `room_y_min` = -2.0 m, measured 2 m left of tag 10's centre.
  Tags are textured tag36h11 quads, orientation verified against the real detector.
- **Sensing**: a tag counts only if it is inside the Tello camera's view (FOV, range, view
  angle, pixel size, dropout, 0.2 s video delay, link loss). A detectable tag decodes 90% of frames up to
  10 deg off its face, falling to 0 at 62 deg (`tag_sensor.view_full_angle` / `max_view_angle`); motion blur
  drops whole frames: 1.0 per rad/s of yaw rate above 0.2 (max 0.45) plus 0.05 per m/s above 0.5, total
  max 0.5 (`tag_sensor.blur_*`). View angle fitted to flights 030-034 + 040-043, blur to 030-034. Observations are the positional
  feed only, never ground truth.
- **Kalman filter** (`env.kalman`, on by default; `env.kalman.enabled=False` trains on the raw
  fix): the observed position is the estimate of `drone_deployment_infra/core/positional_kalman_filter.py`,
  the laptop's file, loaded by path (`tags/kalman.py`) and run on torch for every drone. Constant
  velocity per axis, outlier gate, holds when lost, restarts on the first fix after. Deploy with
  `FilteredPositioner` iff the policy was trained with it. Self-test: `python3
  drone_deployment_infra/core/positional_kalman_filter.py` (pass: `SELFTEST PASSED`).
- **Motion**: the integer sticks go through the deadband and gains in
  `source/action_space_kit/action_space_kit/control/tello_motion_fit.yaml`. Deploy by sending
  `env.last_rc` (the same rounding) as `rc a b c d`, and `land` once the waypoint is reached (auto land).
- **Command rate**: one policy step = one rc packet, 74 ms (`env.command_rate_hz` = 500/37 =
  13.5 Hz) with 0-16 ms arrival jitter. Measured in flight: `recordings/027_TELLO-1.train.jsonl`
  sent at 13.43 Hz (median 70 ms, 80 ms cruising, 55-110 ms), because each tick runs a full-frame
  tag detection before its `stop.wait(0.05)`. (`drone_deployment_tests/rc_rate_bench.py` on the
  desktop gave 19.94 Hz for the older engine; don't use that number.) Per episode each drone loses
  0-15% of rc packets (holds the last rc) and video frames (no fix): the flight reports lost
  3-19% of frames per flight.
- **Motion fit**: `tello_motion_fit.yaml` holds PRIORS until real logs are fitted. Copy the
  laptop's `drone_deployment_infra/recordings/*.train.jsonl` here (logs from before the
  unified layout: `drone_deployment_infra/core/migrate_outputs.py` moves `outputs/` into `recordings/`), then:

```bash
python -u scripts/fit_tello_motion.py --selftest                                  # pass: SELFTEST PASSED
python -u scripts/fit_tello_motion.py          # every recordings/*.train.jsonl; rewrites the yaml
```
  Read its two cross-checks: the rc-vs-vel correlation should be diagonal, and the tag-vs-state
  speed scale should be ~1 (a ~10x or ~0.1x means the engine's vgx/vgy units/axes are wrong;
  then fit with `--speed_source tag`).

```bash
python -u scripts/check_room_env.py --headless --test all > outputs/check_room.log 2>&1
grep -E "^\s*\[(ok|FAIL)|^== |zero |scripted " outputs/check_room.log
python -u scripts/check_room_render.py > outputs/check_room_render.log 2>&1        # real localizer on sim renders
python -u scripts/render_room_overview.py --out outputs/room_overview.png > outputs/room_overview.log 2>&1
```
Pass: every line `[ok]` (66; the Peer task: 35); `zero` 0% success, with landing / no-tag losses equal to -0.5 x
the off-spot / tag-less drone-steps. Render check: every view `[ok]`; a
`[WARN real localizer: ambiguous pose flip]` line is a finding about `positional.py`, not a sim failure.
Train (detached; central, ~27 s per iteration at 512 envs on the RTX 5080, 1500 iterations ~11 h):
`setsid nohup python -u scripts/sb3/train.py --task AS-Tello-Room-v0 --headless --num_envs 512 --max_iterations 1500 > outputs/train_room.log 2>&1 &`.
Watch it from anywhere: `dashboard/serve.sh` prints a public TensorBoard URL (see `dashboard/README.md`).
Live view: `scripts/stream_room.py` flies the newest checkpoint (hot-reloaded every 5 min) with Isaac Sim's
WebRTC livestream (`livestream=2`, port 49100); connect NVIDIA's Isaac Sim WebRTC Streaming Client to the
desktop over Tailscale (Cloudflare quick tunnels can't carry WebRTC). `sb3/play.py --livestream` fails
(Kit's numpy 1.26 can't unpickle the checkpoints), so it exports to TorchScript first like the recorder.

Video (the core recorder for this task; works mid-training: with no `--checkpoint` it takes the
newest `model_N_steps.zip` of the newest run):
```bash
python -u scripts/record_room.py --episodes 2 --out outputs/videos/room_policy.mp4 > outputs/record_room.log 2>&1
grep "\[rec\]" outputs/record_room.log
```
Pass: one `[rec] episode N: ...` line per attempt and `[rec] wrote outputs/videos/room_policy.mp4 (...)`.
Real time (one frame per rc command). `--view overview|side|top`, `--pilot zero` (scene check without
a policy), `--no_dr`, `--set key=value`. Per drone one color: beacon = true position, cube = published
fix, translucent sphere = its waypoint, disc = its landing spot; the bottom panel shows each drone's
phase, rc and fix age. Checkpoint export (VecNormalize + deterministic policy -> TorchScript) is
shared with `record_approach.py` in `action_space_kit/rl/policy_export.py`.

### Single-drone variant: `AS-Tello-Room-Single-v0` (reward tuning)

The room task with **one** drone (`tello_room_single_env_cfg.py`, agent cfg `sb3_ppo_single_cfg.yaml`,
same PPO settings): same room, sensing, rc actions, dynamics randomization and reward terms, but the
drone **starts on the floor** (`start_on_floor`, motors off, same start box) and the episode opens with
the SDK `takeoff` the engine sends before any rc: the firmware climbs at 0.5 m/s to 0.8-1.0 m
(`takeoff_height`) ignoring sticks and land, then the policy flies. Takeoff steps pay no closer reward
and no no-tag loss (the policy isn't flying); the landing loss runs. It gets one of the four spots at
random (`spot_count`). Observation 25 = 24 + taking-off flag. Success: landed within 0.3 m of its spot.
```bash
python -u scripts/check_room_env.py --headless --task AS-Tello-Room-Single-v0 > outputs/check_room_single.log 2>&1
grep -E "^\s*\[(ok|FAIL)|^== |zero " outputs/check_room_single.log    # pass: every line [ok] (36)
setsid nohup python -u scripts/sb3/train.py --task AS-Tello-Room-Single-v0 --headless --num_envs 512 --max_iterations 1500 > outputs/train_room_single.log 2>&1 &
python -u scripts/record_room.py --task AS-Tello-Room-Single-v0 --out outputs/videos/room_single.mp4 > outputs/record_room_single.log 2>&1
```
Reward knobs are the `rew_*` fields in `tello_room_env_cfg.py`; override per run with `env.<field>=<value>`.
