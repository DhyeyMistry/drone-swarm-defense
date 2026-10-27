#!/usr/bin/env bash
# Rebuild the Action Space training environment on a fresh Ubuntu 24.04 box with an
# NVIDIA GPU. Installs Isaac Sim 5.1 + Isaac Lab 2.3.2 + this kit into ISAAC_ROOT.
#
#   ./setup.sh [ISAAC_ROOT]      # default: /mnt/data/isaac
#
# Traps this script exists to avoid, both hit during the reference build:
#   * flatdict 4.0.1 fails to build with setuptools >= 81, and Isaac Lab's installer
#     reports the error but carries on -- leaving the core isaaclab package missing.
#   * an automatic kernel upgrade can leave the NVIDIA driver unbuilt, so the GPU
#     disappears after a reboot. Check nvidia-smi before blaming anything else.
#   * an active conda env (even `base`) makes isaaclab.sh use conda's python instead of
#     the venv's; its vscode step and convert.sh then fail. The script unsets conda vars.
# Safe to re-run: finished steps are skipped or are quick no-op reinstalls.
set -euo pipefail

# resolve before any cd: $0 may be relative (./setup.sh)
KIT_DIR=$(cd "$(dirname "$0")" && pwd)

ISAAC_ROOT=${1:-/mnt/data/isaac}
ISAACLAB_VERSION=${ISAACLAB_VERSION:-v2.3.2}
ISAACSIM_VERSION=${ISAACSIM_VERSION:-5.1.0}

echo "== checking GPU"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || {
  echo "ERROR: no working NVIDIA driver. Fix that first (see docs/shared-server.md)." >&2
  exit 1
}

echo "== system packages"
APT_PKGS=(cmake build-essential git git-lfs curl ffmpeg libglu1-mesa libvulkan1 vulkan-tools)
MISSING=()
for p in "${APT_PKGS[@]}"; do
  dpkg -s "$p" >/dev/null 2>&1 || MISSING+=("$p")
done
if [ ${#MISSING[@]} -gt 0 ]; then
  echo "installing: ${MISSING[*]}"
  sudo DEBIAN_FRONTEND=noninteractive apt-get update -qq
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y "${MISSING[@]}"
else
  echo "all present, skipping apt (no sudo needed)"
fi

echo "== workspace: $ISAAC_ROOT"
mkdir -p "$ISAAC_ROOT"
cd "$ISAAC_ROOT"
export UV_CACHE_DIR="$ISAAC_ROOT/.cache/uv" PIP_CACHE_DIR="$ISAAC_ROOT/.cache/pip"
export TMPDIR="$ISAAC_ROOT/tmp" UV_PYTHON_INSTALL_DIR="$ISAAC_ROOT/.python"
mkdir -p "$TMPDIR"
export OMNI_KIT_ACCEPT_EULA=YES   # NVIDIA Omniverse EULA: https://docs.omniverse.nvidia.com/eula/

command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

echo "== python 3.11 environment"
# isaaclab.sh picks $CONDA_PREFIX's python over $VIRTUAL_ENV's, so an active conda env
# (even `base`) silently hijacks it; the vscode step and convert.sh then fail. Drop it.
unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_SHLVL CONDA_PYTHON_EXE CONDA_EXE
[ -d env_isaaclab ] || uv venv --python 3.11 --seed env_isaaclab
source env_isaaclab/bin/activate

echo "== Isaac Sim $ISAACSIM_VERSION"
pip install -q --upgrade pip
pip install "isaacsim[all,extscache]==${ISAACSIM_VERSION}.0" --extra-index-url https://pypi.nvidia.com

echo "== PyTorch (CUDA 12.8)"
pip install -q -U torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128

echo "== Isaac Lab $ISAACLAB_VERSION"
[ -d IsaacLab ] || git clone -q https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab && git checkout -q "$ISAACLAB_VERSION"

# work around the flatdict build failure before it can silently skip isaaclab
uv pip install -q "setuptools<81" wheel
uv pip install -q --no-build-isolation flatdict==4.0.1

./isaaclab.sh --install
uv pip install -q --editable source/isaaclab   # belt and braces: the step above can skip it
cd ..

echo "== this kit"
uv pip install -q -e "$KIT_DIR/source/action_space_kit"
uv pip install -q djitellopy   # real-drone bridge

echo "== Tello asset"
if [ ! -f "$ISAAC_ROOT/assets/tello.usd" ]; then
  mkdir -p "$ISAAC_ROOT/assets"
  ISAACLAB="$ISAAC_ROOT/IsaacLab" "$KIT_DIR/assets/tello/convert.sh" "$ISAAC_ROOT/assets/tello.usd"
fi

echo "== verifying"
python -c "import isaaclab, isaacsim, torch; print('isaaclab ok, cuda:', torch.cuda.is_available())"
cd "$KIT_DIR" && python scripts/list_envs.py 2>/dev/null | grep -E "AS-" || {
  echo "ERROR: kit tasks did not register" >&2; exit 1; }

cat <<EOF

Done. To train:

  cd $KIT_DIR
  source $ISAAC_ROOT/env_isaaclab/bin/activate
  export OMNI_KIT_ACCEPT_EULA=YES TELLO_USD_PATH=$ISAAC_ROOT/assets/tello.usd
  python scripts/skrl/train.py --task AS-Tello-Hover-v0 --headless --num_envs 1024 --max_iterations 200
EOF
