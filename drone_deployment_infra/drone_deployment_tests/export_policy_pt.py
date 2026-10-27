"""Export a room checkpoint (model.zip + VecNormalize) to a self-contained TorchScript folder.

    python export_policy_pt.py                       # qualified 4-drone policy -> <its folder>/pt/
    python export_policy_pt.py --checkpoint RUN/model_N_steps.zip --out some_dir

The output folder holds policy.pt plus the checkpoint's params/ and assets/ (manifest, room
snapshot, motion fit), so a laptop can run the policy with torch only: no stable-baselines3
and no custom policy classes (load it with torch.jit.load; see core/room_policy.py for the
observation and landing gate it expects).

policy.pt maps a NORMALIZED frame-stacked history [N, obs_dim] to the deterministic action
[N, act_dim], clipped to the action space exactly as PPO.predict does. The per-frame
normalization stats ride along as buffers (obs_mean, obs_var, clip_obs, epsilon) because the
adapter normalizes each frame before stacking (SB3 zeroes absent frames AFTER normalization).
Run on the desktop (needs stable-baselines3); it checks the export against PPO.predict.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "drone_deployment_infra"))
sys.path.insert(0, str(REPO / "drone_deployment_infra/core"))
from room_policy import RoomPolicy  # noqa: E402

DEFAULT_CHECKPOINT = REPO / "simulation_infra/policies/peer_qualified_84M/model.zip"


class Exported(torch.nn.Module):
    def __init__(self, model, normalizer):
        super().__init__()
        self.policy = model.policy.eval()
        f32 = lambda a: torch.as_tensor(np.asarray(a, dtype=np.float32))  # noqa: E731
        dim = model.observation_space.shape[0]
        if normalizer is not None and normalizer.norm_obs:
            mean, var, clip, eps = normalizer.obs_rms.mean, normalizer.obs_rms.var, normalizer.clip_obs, normalizer.epsilon
        else:
            mean, var, clip, eps = np.zeros(1), np.ones(1), np.inf, 0.0
        self.register_buffer("obs_mean", f32(mean))
        self.register_buffer("obs_var", f32(var))
        self.register_buffer("clip_obs", f32(clip))
        self.register_buffer("epsilon", f32(eps))
        self.register_buffer("obs_dim", torch.tensor(dim))
        self.register_buffer("act_dim", torch.tensor(model.action_space.shape[0]))
        self.register_buffer("low", f32(model.action_space.low))
        self.register_buffer("high", f32(model.action_space.high))

    def forward(self, history):
        return torch.maximum(torch.minimum(self.policy._predict(history, deterministic=True), self.high), self.low)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    ap.add_argument("--out", help="output folder (default: <checkpoint folder>/pt)")
    args = ap.parse_args()
    ckpt = Path(args.checkpoint).resolve()
    out = Path(args.out).resolve() if args.out else ckpt.parent / "pt"

    ref = RoomPolicy(ckpt)  # same loading + contract checks as deployment
    model, norm = ref.model, ref.normalizer
    module = Exported(model, norm)
    dim = model.observation_space.shape[0]
    with torch.no_grad():
        traced = torch.jit.trace(module, torch.zeros(2, dim))

    # Check against PPO.predict on normalized-scale inputs, including stacks with zero padding.
    rng = np.random.default_rng(0)
    x = rng.normal(0.0, 1.5, (4096, dim)).astype(np.float32)
    x[::7, : dim // 2] = 0.0
    x = np.clip(x, -module.clip_obs.item(), module.clip_obs.item())
    want, _ = model.predict(x, deterministic=True)
    with torch.no_grad():
        got = traced(torch.from_numpy(x)).numpy()
    err = float(np.abs(got - want).max())
    print(f"export check: max |pt - PPO.predict| = {err:.2e} over {len(x)} inputs")
    if err > 1e-5:
        raise SystemExit("EXPORT FAILED: TorchScript output differs from PPO.predict")

    out.mkdir(parents=True, exist_ok=True)
    traced.save(str(out / "policy.pt"))
    for sub in ("params", "assets"):
        if (ckpt.parent / sub).is_dir():
            shutil.copytree(ckpt.parent / sub, out / sub, dirs_exist_ok=True)
    m = re.match(r"model_(\d+)_steps$", ckpt.stem)
    info = {
        "source_checkpoint": str(ckpt),
        "source_sha256": hashlib.sha256(ckpt.read_bytes()).hexdigest(),
        "normalizer": f"model_vecnormalize_{m[1]}_steps.pkl" if m else "model_vecnormalize.pkl",
        "policy_sha256": hashlib.sha256((out / "policy.pt").read_bytes()).hexdigest(),
        "obs_dim": dim, "act_dim": int(model.action_space.shape[0]), "frame_stack": ref.n_stack,
        "max_abs_error_vs_ppo": err,
    }
    (out / "export.json").write_text(json.dumps(info, indent=2) + "\n")

    # Reload as the laptop will (torch only) and re-check.
    loaded = torch.jit.load(str(out / "policy.pt"), map_location="cpu")
    with torch.no_grad():
        err2 = float(np.abs(loaded(torch.from_numpy(x)).numpy() - want).max())
    print(f"reloaded check: {err2:.2e}")
    print(f"EXPORT PASSED: {out / 'policy.pt'}")


if __name__ == "__main__":
    main()
