# Sharing the GPU box

The reference server is an AWS g5.2xlarge: one NVIDIA A10G (24 GB), 8 vCPUs, 30 GB RAM.
One training run at 1024 environments uses roughly 3-5 GB of GPU memory, so several teams
can train at once -- but only if everyone sizes their runs sensibly.

## Rules of thumb for 4-6 teams

| Task type | `--num_envs` per team | Notes |
|---|---|---|
| Hover / waypoint | 512-1024 | ~3-5 GB, trains in minutes |
| Defend (state-based) | 256-512 | 4 drones per env costs more than it looks |
| Camera-based | 32-64 | rendering dominates; book a slot rather than sharing |

Check what's running before you start:

```bash
nvidia-smi                      # who is using the GPU right now
```

## Keep runs alive across disconnects

Training dies with your SSH session unless you detach it:

```bash
tmux new -s myteam              # start a named session
# ... launch training ...
# Ctrl-B then D to detach; log out freely
tmux attach -t myteam           # come back later
```

## Per-team directories

Work inside your own directory so runs and checkpoints do not collide:

```bash
cp -r /mnt/data/isaac/action_space_kit /mnt/data/isaac/teams/<your-team>
```

Keep everything on `/mnt/data` (100 GB). The system disk has under 8 GB free and Isaac Sim
caches are large.

## If the GPU is full

`torch.OutOfMemoryError` means someone else's run plus yours exceeded 24 GB. Halve
`--num_envs`, or wait. Never kill another team's process.
