# Sim-to-sim: coordinated pursuit + PPO residual (MuJoCo → Isaac Lab)

> 🚧 **More coming soon:** the MuJoCo environment, the training and evaluation code, the plots and the simulation video will be added to this folder. The numbers below are the results presented to the judges.

![MuJoCo: coordinator + chase, 3 defenders vs 2 attackers](../../docs/media/presentation/mujoco_coordinator_chase.jpg)

*MuJoCo, coordinator + chase, 3 defenders vs 2 attackers: the base (green) with attackers approaching. This is a still from the simulation video shown in the presentation.*

## Approach

- **Coordinated defense:** one interceptor per attacker, plus a reserve guarding the base.
- **Lead-pursuit control**, with an optional **PPO residual** on top and a **safety shield**.
- **Built in MuJoCo, validated unchanged in Isaac Lab** (PhysX), on the kit's `AS-Defend-v0` task ([`simulation_infra/`](../../simulation_infra/)). Along the way we fixed a force-frame bug in the provided environment.
- **Deployed on real Tellos**, using onboard camera localization and shared telemetry. The deployed system is in [`drone_deployment_infra/our_defense/`](../../drone_deployment_infra/our_defense/).

## Results (Isaac Lab `AS-Defend-v0`, 200 episodes per row, 20 Hz, headless on an H100)

| Scenario | Greedy chase (provided baseline) | **Ours: coordinator + chase** | Ours + RL residual |
|---|---|---|---|
| 2 vs 2, straight attackers | 89.0% breach (55% intercepted) | **0.0% breach (100% intercepted)** | 24.0% breach (87%) |
| 2 vs 2, weaving attackers | 78.5% breach (61%) | **0.0% breach (100%)** | 19.0% breach (90%) |
| 3 vs 2, straight attackers | 82.5% breach (58%) | **0.0% breach (100%)** | 7.5% breach (96%) |
| 3 vs 2, weaving attackers | 75.5% breach (62%) | **0.0% breach (100%)** | 4.5% breach (98%) |

- **0% breaches over 800 episodes** for our coordinated defense, in every scenario. The provided greedy-chase baseline breaches **75–89%** of the time.
- **With realistic sensing** (3 vs 2 weaving, 0.2 s delay, noisy positions, dropouts), the breach rate is **7.5%**.
- **Why coordination wins:** the baseline sends every defender after the nearest attacker, so the other attacker gets through. We assign one interceptor per attacker, plus a reserve guarding the base.
- **Same code, new engine:** developed in MuJoCo and run unchanged in Isaac Lab. The RL residual did not generalize to the new layout, so **pursuit + coordination is the core**, and it's what the real-drone system uses.

## To be added

- [ ] MuJoCo environment and the coordinator / lead-pursuit controller
- [ ] PPO residual training code, configs and checkpoints
- [ ] Isaac Lab validation scripts, including the `AS-Defend-v0` force-frame fix
- [ ] Evaluation scripts and raw results for the table above
- [ ] Plots and the simulation video
