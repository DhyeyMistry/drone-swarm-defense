#!/bin/bash
# usage: scripts/run_iter.sh <name> <max_iterations> [hydra overrides...]
# Trains AS-Tello-Approach-v0 (512 envs, lr 1e-3) then scores the final model.
# Logs and a one-line result per run go to $LOOP_DIR (default: <kit>/outputs/policy_loop).
# SPAWN_AZ=<rad> overrides the spawn direction range (default 0.0 = always +x).
# Launch detached so it survives an SSH drop:
#   setsid nohup scripts/run_iter.sh it7 800 env.rew_time=-0.05 < /dev/null > /dev/null 2>&1 &
set -u
NAME=$1; ITERS=$2; shift 2
KIT=$(cd "$(dirname "$0")/.." && pwd)
ISAAC_ROOT=${ISAAC_ROOT:-$(dirname "$KIT")}
cd "$KIT"
source "$ISAAC_ROOT/env_isaaclab/bin/activate"
export OMNI_KIT_ACCEPT_EULA=YES TELLO_USD_PATH=${TELLO_USD_PATH:-$ISAAC_ROOT/assets/tello.usd}
L=${LOOP_DIR:-$KIT/outputs/policy_loop}
mkdir -p "$L"
python scripts/sb3/train.py --task AS-Tello-Approach-v0 --headless --num_envs 512 \
  --max_iterations $ITERS agent.learning_rate=1e-3 env.spawn_azimuth_range=${SPAWN_AZ:-0.0} "$@" > $L/$NAME.train.log 2>&1
RUN=$(ls -td logs/sb3/AS-Tello-Approach-v0/*/ | head -1)
echo "$NAME $RUN $*" >> $L/runs.txt
python scripts/sb3/play.py --task AS-Tello-Approach-v0 --headless --num_envs 64 --episodes 256 \
  --checkpoint ${RUN}model.zip env.spawn_azimuth_range=${SPAWN_AZ:-0.0} "$@" > $L/$NAME.eval.log 2>&1
echo "== $NAME ($RUN) overrides: $*" >> $L/results.txt
grep "episodes\]" $L/$NAME.eval.log | tail -1 >> $L/results.txt
tail -2 $L/results.txt
