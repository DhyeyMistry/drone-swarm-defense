# Approaches

Team 5 built three ways to defend the table. They share the same goal and the same Tellos, but each one stresses a different part of the problem.

| Approach | Where it is | Status |
|---|---|---|
| **AprilTag localization + telemetry pursuit**: flown at the competition | [`drone_deployment_infra/our_defense/`](../drone_deployment_infra/our_defense/) | ✅ code, tests, real-flight and simulation results in this repo |
| **Sim-to-sim: coordinated pursuit + PPO residual**: MuJoCo → Isaac Lab | [`sim_to_sim_rl/`](sim_to_sim_rl/) | 📝 described; code, plots and video coming soon |
| **Visual servo defense**: camera-only detection, tracking and forecasting | [`visual_servo/`](visual_servo/) | 📝 described; code, plots and video coming soon |

How they fit together:
- **The sim-to-sim work** checked the *strategy* at scale: one interceptor per attacker plus a reserve guarding the base. It found that coordination, not the learned residual, is what drives the breach rate to zero. That same coordination logic is what the deployed system flies.
- **The visual servo work** removes the dependence on published attacker positions. Each defender finds and tracks the attackers with its own camera, which matters if the organizers' telemetry is missing or late.
