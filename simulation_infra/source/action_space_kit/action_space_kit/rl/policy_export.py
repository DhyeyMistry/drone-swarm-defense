"""Find SB3 checkpoints and export one (with its VecNormalize stats) as TorchScript.

The recorders run Isaac Sim in livestream mode, which loads omni.kit.pip_archive: its
prebundled numpy 1.26 shadows the venv's numpy 2, so the numpy-2 pickles inside model.zip /
*.pkl can't be loaded in that process. ``export_policy`` loads them in a clean interpreter
and writes a TorchScript module the recorder can run on any device.

Only the standard library is imported here.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

_EXPORT = """
import os, pickle, sys, numpy as np, torch
from stable_baselines3 import PPO
ckpt, norm_path, out, n_stack = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
model = PPO.load(ckpt, device="cpu")
dim = model.observation_space.shape[0] // n_stack
mean, var, clip, eps = np.zeros(dim), np.ones(dim), np.inf, 0.0
if os.path.exists(norm_path):
    v = pickle.load(open(norm_path, "rb"))
    if v.norm_obs:
        mean, var, clip, eps = v.obs_rms.mean, v.obs_rms.var, v.clip_obs, v.epsilon
else:
    raise FileNotFoundError(f"Missing checkpoint normalization statistics: {norm_path}")
f32 = lambda a: torch.as_tensor(np.asarray(a, dtype=np.float32))


class Exported(torch.nn.Module):
    # VecNormalize (per frame, before VecFrameStack) -> deterministic policy -> action clip, as in PPO.predict.
    def __init__(self):
        super().__init__()
        self.policy = model.policy.eval()
        self.register_buffer("mean", f32(np.tile(mean, n_stack)))
        self.register_buffer("std", f32(np.sqrt(np.tile(var, n_stack) + eps)))
        self.register_buffer("low", f32(model.action_space.low))
        self.register_buffer("high", f32(model.action_space.high))
        self.clip = float(clip)
        self.n_stack = n_stack
        self.frame_dim = dim

    def forward(self, obs):
        raw = obs
        obs = torch.clamp((raw - self.mean) / self.std, -self.clip, self.clip)
        if self.n_stack > 1:
            # Recorders stack raw observations. SB3 stacks normalized observations, with
            # absent history set to zero AFTER normalization. Room/approach frames contain
            # a unit heading/quaternion, so an all-zero frame uniquely denotes padding.
            padding = (raw.reshape(raw.shape[0], self.n_stack, self.frame_dim) == 0).all(-1)
            obs = torch.where(padding.repeat_interleave(self.frame_dim, dim=-1), torch.zeros_like(obs), obs)
        return torch.maximum(torch.minimum(self.policy._predict(obs, deterministic=True), self.high), self.low)


with torch.no_grad():
    torch.jit.trace(Exported(), torch.zeros(2, model.observation_space.shape[0])).save(out)
"""


def norm_path_for(ckpt: str | Path) -> Path:
    """VecNormalize stats saved next to a checkpoint: model_N_steps.zip -> model_vecnormalize_N_steps.pkl."""
    ckpt = Path(ckpt)
    m = re.match(r"model_(\d+)_steps$", ckpt.stem)
    return ckpt.with_name(f"model_vecnormalize_{m.group(1)}_steps.pkl" if m else "model_vecnormalize.pkl")


def latest_checkpoint(task_log_dir: str | Path) -> Path:
    """Newest run under logs/sb3/<task>: its final model.zip, or (mid-training) its newest model_N_steps.zip."""
    runs = sorted((p for p in Path(task_log_dir).iterdir() if p.is_dir()), key=lambda p: p.stat().st_mtime)
    for run in reversed(runs):
        if (run / "model.zip").exists():
            return run / "model.zip"
        steps = [(int(m.group(1)), p) for p in run.glob("model_*_steps.zip") if (m := re.match(r"model_(\d+)_steps$", p.stem))]
        if steps:
            return max(steps)[1]
    raise FileNotFoundError(f"no checkpoint under {task_log_dir}")


def export_policy(ckpt: str | Path, n_stack: int, out: str | Path) -> Path:
    """Write ``out``: TorchScript obs (N, n_stack * dim) -> deterministic action (N, act_dim)."""
    subprocess.run([sys.executable, "-I", "-c", _EXPORT, str(ckpt), str(norm_path_for(ckpt)), str(out), str(n_stack)],
                   check=True)
    return Path(out)
