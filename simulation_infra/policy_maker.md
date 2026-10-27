# Policy maker: tuning rewards and training for `AS-Tello-Approach-v0`

How to train your own policy for the 4-drone approach-and-land task: where every reward
lives, what each knob does, how many simulations run per batch, and the command to run.
It assumes you've finished the install in `ACTION_SPACE_KIT.md` (or `KIT_SETUP_GUIDE.md`).

Measured on the reference machine (RTX 5080 16 GB, 2026-09-27) unless noted.

## TL;DR: the command

Open a shell (this is needed in every new terminal):

```bash
conda deactivate 2>/dev/null; conda deactivate 2>/dev/null
cd ~/Action_Space
source ~/isaac/env_isaaclab/bin/activate
export OMNI_KIT_ACCEPT_EULA=YES TELLO_USD_PATH=~/isaac/assets/tello.usd
```

Train. This is the recipe that produced the shipped policy; tweak the `env.*` / `agent.*` values:

```bash
python -u scripts/sb3/train.py --task AS-Tello-Approach-v0 --headless \
  --num_envs 512 --max_iterations 800 \
  agent.learning_rate=1e-3 \
  env.spawn_azimuth_range=0.0 \
  env.rew_scale_separation=-20.0 env.rew_violation=-100.0
```

That's ~4.7 s per iteration, so **800 iterations take about an hour** at 512 sims. The result
lands in `logs/sb3/AS-Tello-Approach-v0/<date_time>/model.zip`.

Score it. Pass the **same `env.*` overrides** you trained with:

```bash
python -u scripts/sb3/play.py --task AS-Tello-Approach-v0 --headless --num_envs 64 --episodes 256 \
  --checkpoint logs/sb3/AS-Tello-Approach-v0/<date_time>/model.zip \
  env.spawn_azimuth_range=0.0 env.rew_scale_separation=-20.0 env.rew_violation=-100.0
```

The last line is the score. The bar to beat is the shipped policy: `near_base_frac 0.897,
success_rate 0.864, collision_rate 0.136`.

Or do both in one go and append the score to `outputs/policy_loop/results.txt`:

```bash
ISAAC_ROOT=~/isaac scripts/run_iter.sh my_run 800 env.rew_scale_separation=-20.0 env.rew_violation=-100.0
```

(`run_iter.sh` hard-codes 512 sims, lr 1e-3 and `spawn_azimuth_range=0.0`, and you must set
`ISAAC_ROOT`.)

---

## 1. Two ways to change anything

| Way | When | Example |
|---|---|---|
| **Command-line override** (Hydra) | experiments; nothing to edit, and it's recorded with the run | `env.rew_gate=150.0`, `agent.n_epochs=10`, `env.dynamics.domain_randomization=False`, `'env.dynamics.latency_steps=[0,1]'` |
| **Edit the config file** | changing the default for everyone | `tello_approach_env_cfg.py`, `agents/sb3_ppo_cfg.yaml` |

- `env.<name>` means any field of `TelloApproachEnvCfg`, and `env.dynamics.<name>` any field of
  `TelloDynamicsCfg`. `agent.<name>` means any key in `sb3_ppo_cfg.yaml`.
- Quote overrides containing brackets: `'env.dynamics.latency_steps=[0,1]'`.
- Every run writes what it actually used to `logs/.../<run>/params/env.yaml`,
  `params/agent.yaml` and `command.txt`. Check there if you're unsure whether an override applied.

All paths below are relative to `source/action_space_kit/action_space_kit/`.

## 2. Where the rewards are

- **Values** (the numbers you tune): `tasks/direct/tello_approach/tello_approach_env_cfg.py`, lines 99-112.
- **Formula** (how they combine): `_get_rewards()` in `tasks/direct/tello_approach/tello_approach_env.py`, line 307.

Each drone gets its own reward every policy step (20 Hz):

```
reward = active * (progress + rates * dt + rew_time + events) + rew_violation * violated
  progress = rew_closer if the drone ended the step closer to its target, else rew_not_closer
  rates    = rew_scale_barrier*barrier + rew_scale_separation*separation
             + rew_scale_effort*|action|^2 + rew_scale_action_rate*|Δaction|^2
  events   = (rew_gate + rew_scale_entry_speed*inward_speed) on the step it reaches its gate
             + rew_goal on the step it reaches its landing spot
  active   = 0 once a drone is landing/landed (it stops earning), else 1
```

"Target" means the drone's own gate waypoint, then (after the gate) the point 0.5 m above its own landing pad.

| Knob (`env.`…) | Default | Shipped policy | What it rewards / punishes | Turn it up if… |
|---|---|---|---|---|
| `rew_closer` | +1.0 | +1.0 | each step that ends closer to the current target | drones dawdle |
| `rew_not_closer` | -0.5 | -0.5 | each step that doesn't (further, or equal) | drones hover in place |
| `rew_time` | -0.05 | -0.05 | every step until the drone arrives | episodes are slow (`duration_s` high) |
| `rew_gate` | +100 | +100 | reaching its own gate waypoint (once) | `gates_passed` < 4 |
| `rew_goal` | +100 | +100 | reaching its landing spot (once; then it auto-lands) | drones pass the gate but don't land |
| `rew_scale_entry_speed` | 0.0 | 0.0 | inward speed at the moment of passing the gate | you want faster entries (off by default) |
| `rew_violation` | -30 | **-100** | one-off, and **ends the episode**: entering the sphere without its gate, drone-drone contact (< `collision_distance` 0.15 m), below 0.2 m / above 8 ft, > 13 m out, tilt > 60° | `collision_rate`, `illegal_entry_rate` or `crash_rate` are high |
| `rew_scale_barrier` | -4.0 /s | -4.0 | hugging the keep-out sphere (within `barrier_margin` 0.5 m) away from its gate, before the gate is passed | illegal entries |
| `rew_scale_separation` | -4.0 /s | **-20.0** | being closer than `separation_distance` (0.3 m) to another drone, scaled by how close | collisions (this was the big fix for the shipped policy) |
| `rew_scale_effort` | -0.01 /s | -0.01 | squared stick input | jerky, full-throttle flying |
| `rew_scale_action_rate` | -0.05 /s | -0.05 | change in stick input step to step | twitchy flying (matters for real Tellos) |

Rules of thumb:
- Scale matters relative to the others. One gate or goal (+100) equals ~100 steps of progress. A violation
  must clearly outweigh what a drone could still gain, or it will trade a crash for a shortcut.
- The `/s` terms are multiplied by `dt` = 0.05 s per step, so -20 /s costs -1 per step at full proximity.
- Set any safety term to 0 to switch it off. With all of them at 0 you get the bare
  +1 / -0.5 / +100 / +100 structure.
- Change one or two things per run and compare against a baseline run. With the random seed
  alone, runs vary by a few percentage points.

## 3. Batch size: how many sims per batch

The shipped policy is **one network shared by all 4 drones**. `rl/sb3_shared.py` turns each drone of
each simulated environment into its own row, so:

| Quantity | Formula | At the defaults |
|---|---|---|
| parallel sims (`--num_envs`) | you choose | 512 |
| drones / SB3 rows | `num_envs × 4` | 2048 |
| transitions per iteration (rollout) | `n_steps × num_envs × 4` | 64 × 512 × 4 = **131,072** |
| minibatch size | rollout ÷ `n_minibatches` | 131,072 ÷ 4 = 32,768 |
| gradient steps per iteration | `n_epochs × n_minibatches` | 5 × 4 = 20 |
| total training steps | `max_iterations × rollout` | 800 × 131,072 ≈ 105 M |

`--max_iterations` is the one to set. It overrides `n_timesteps` in the YAML.

**Measured on the RTX 5080 (16 GB):**

| `--num_envs` | drones | GPU memory | time / iteration | 800 iterations |
|---|---|---|---|---|
| 512 | 2,048 | small | ~4.7 s | ~1 h |
| 2048 | 8,192 | 4.2 GB | ~3.7 s* | ~50 min* |
| 4096 | 16,384 | 5.5 GB | ~7.5 s | ~1.7 h |

*Early iterations only, from a 2-iteration test; treat as approximate.

More sims means a larger, less noisy batch per iteration, but it's **not** a free speed-up: the
rollout size changes, so the effective learning rate and number of updates change too. The
shipped recipe was tuned at 512. If you go to 2048+, keep `max_iterations` similar (each
iteration now sees 4× the data) and try a lower `agent.learning_rate` (e.g. 3e-4) if training
gets unstable. Compare against a 512 run before trusting it.

## 4. PPO settings (`agent.…`)

File: `tasks/direct/tello_approach/agents/sb3_ppo_cfg.yaml`

| Key | Default | Notes |
|---|---|---|
| `learning_rate` | 3e-4 | the shipped recipe uses **1e-3** (via `agent.learning_rate=1e-3`) |
| `n_steps` | 64 | steps per env per rollout (64 × 0.05 s = 3.2 s of flight) |
| `n_minibatches` | 4 | converted to `batch_size` automatically (see §3) |
| `n_epochs` | 5 | passes over each rollout |
| `gamma` / `gae_lambda` | 0.99 / 0.95 | discount / advantage smoothing |
| `ent_coef` | 0.005 | exploration bonus; raise if the policy collapses to hovering |
| `clip_range` / `target_kl` | 0.2 / 0.02 | PPO trust region; `target_kl` stops an update early |
| `policy_kwargs.net_arch` | [128, 128] | hidden layers (ELU) |
| `policy_kwargs.log_std_init` | -0.5 | initial action noise |
| `frame_stack` | 8 | the policy sees each drone's last 8 observations |
| `seed` | 42 | or pass `--seed N`; change it to check a result isn't luck |

**Changing `net_arch` or `frame_stack` makes old checkpoints incompatible.** A real Tello ground
station has to use the same frame stack.

## 5. Other knobs worth knowing

**Scenario / difficulty** (`env.…`, cfg lines 58-96). Use these for a curriculum: train easy, then step
back to the defaults.

| Knob | Default | Effect |
|---|---|---|
| `spawn_azimuth_range` | π (any side) | 0.0 = always spawn on +x. The shipped policy was trained with **0.0** |
| `spawn_distance` | 30 ft | how far out the drones start |
| `keepout_radius` | 3.0 m | the sphere they must enter through their gate |
| `gate_radius` | 0.4 m | how close counts as passing the gate (bigger = easier) |
| `approach_min_separation` | 10° | how close two gate corridors can be (bigger = fewer collisions) |
| `landing_capture_radius` | 0.25 m | how close to the pad counts as arrived |
| `episode_length_s` | 25 s | time limit |
| `collision_distance` / `separation_distance` | 0.15 / 0.3 m | contact threshold / where the separation penalty starts |

**Flight model randomization** (`env.dynamics.…`, `control/tello_dynamics.py` line 93+). Every drone draws its own
parameters each episode, so the policy transfers to real, imperfect Tellos:

| Knob | Default | Effect |
|---|---|---|
| `domain_randomization` | True | False = nominal drone every time (easier, transfers worse) |
| `latency_steps` | [0, 1, 2] | command delay in 50 ms steps |
| `twr_range`, `mass_range` | (0.95, 1.05) | thrust-to-weight, true vs assumed mass |
| `k_aero_range` | (0.5, 2.0) | rotor drag |
| `kp/ki/kd_omega_range`, `outer_gain_range`, `tau_m_range` | ±15-30% | firmware PID gains, motor lag |

## 6. Watching training

```bash
tensorboard --logdir logs/sb3/AS-Tello-Approach-v0
```

Metrics that tell you what to tune (under `Episode/`):

| Metric | Want | If it's bad, look at |
|---|---|---|
| `success_rate` | ↑ (all 4 landed) | everything below |
| `gates_passed` | → 4.0 | `rew_gate`, `gate_radius` |
| `drones_arrived`, `drones_landed` | → 4.0 | `rew_goal`, `landing_capture_radius` |
| `collision_rate` | ↓ | `rew_scale_separation`, `rew_violation` |
| `illegal_entry_rate` | ↓ | `rew_scale_barrier`, `rew_violation` |
| `crash_rate` | ↓ | `rew_violation`, `rew_scale_effort` |
| `near_base_frac` (play.py) | ≥ 0.80 bar | the headline score |

`rollout/ep_rew_mean` climbs as it learns (it starts around -20 and goes positive within a few iterations),
but judge the policy by the rates above, not reward.

## 7. Long runs, checkpoints, resuming, videos

- Detach long runs so closing the terminal doesn't kill them:
  `setsid nohup python -u scripts/sb3/train.py ... > outputs/train_myrun.log 2>&1 < /dev/null &`
- Checkpoints are saved every 50 iterations (`--save_interval N`) as `model_<steps>_steps.zip`,
  each with its own `model_vecnormalize_<steps>_steps.pkl`. You can score any of them with `play.py`.
- `--checkpoint <model.zip>` on `train.py` resumes the **network weights** only. Observation
  normalization stats start fresh, so expect a short dip at the start.
- Make a video of your policy:
  ```bash
  python -u scripts/record_approach.py --pilot policy --checkpoint <run>/model.zip \
    --spawn_azimuth_range 0.0 --set rew_scale_separation=-20.0 rew_violation=-100.0 \
    --episodes 3 --out outputs/videos/my_policy.mp4 --frame_dir outputs/frames
  ```
  (`record_approach.py` takes env overrides as `--set key=value`, without the `env.` prefix,
  and only for top-level fields; `dynamics.*` can't be set there, except `--no_dr`.)
- To keep a good policy in git, copy `model.zip`, `model_vecnormalize.pkl`, `params/` and
  `command.txt` into `policies/<name>/`, and add a row to `policies/README.md`.
