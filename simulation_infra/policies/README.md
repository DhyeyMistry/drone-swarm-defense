# Trained policies

Small SB3 checkpoints kept in the repo so a new machine can play and score a policy
without retraining. Everything else under `logs/` stays out of git.

| Dir | Task | Trained with | Eval (256 episodes) |
|---|---|---|---|
| `approach_it5_room_fixed` | `AS-Tello-Approach-v0` | 800 iters, 512 envs, `env.rew_scale_separation=-20.0 env.rew_violation=-100.0` (full command in `command.txt`) | near_base_frac 0.897, success 0.864, collision 0.136 |

The room geometry it was trained on: four drones spawn in a 5x5 ft box 30 ft out on +x,
9 ft ceiling (max_height 8 ft), drone-drone contact allowed only when both are landing.
Success bar: `near_base_frac >= 0.80` (share of drones that get within 3 ft of the base).

Score it:

```bash
python scripts/sb3/play.py --task AS-Tello-Approach-v0 --headless --num_envs 64 --episodes 256 \
  --checkpoint policies/approach_it5_room_fixed/model.zip \
  env.spawn_azimuth_range=0.0 env.rew_scale_separation=-20.0 env.rew_violation=-100.0
```

`play.py` loads `model_vecnormalize.pkl` from next to the checkpoint.
