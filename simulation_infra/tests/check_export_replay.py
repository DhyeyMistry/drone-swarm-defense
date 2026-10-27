"""CPU check of exported inference and the offline adapter using captured simulator feeds."""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--replay", default="outputs/policy_replay.jsonl")
parser.add_argument("--out", default="outputs/policy_export_test.pt")
args = parser.parse_args()
repo = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "policy_export_check", repo / "simulation_infra/source/action_space_kit/action_space_kit/rl/policy_export.py"
)
export = importlib.util.module_from_spec(spec)
spec.loader.exec_module(export)
sys.path.insert(0, str(repo / "drone_deployment_infra"))
from core.room_policy import RoomPolicy

adapter = RoomPolicy(args.checkpoint)
policy = torch.jit.load(str(export.export_policy(args.checkpoint, adapter.n_stack, args.out)))
raw_history = np.zeros(adapter.model.observation_space.shape, np.float32)
count = 0
for line in Path(args.replay).read_text().splitlines():
    feed = json.loads(line)
    if feed["reset"]:
        adapter.reset()
        raw_history[:] = 0
    result = adapter.step(feed)
    frame = np.asarray(result["observation"], np.float32)
    raw_history = np.concatenate([raw_history[len(frame) :], frame])
    with torch.inference_mode():
        predicted = policy(torch.from_numpy(raw_history[None])).numpy()
    np.testing.assert_allclose(predicted, feed["expected_action"], atol=2e-5, rtol=1e-4)
    np.testing.assert_allclose(result["action"], feed["expected_action"], atol=2e-5, rtol=1e-4)
    count += 1
print(f"EXPORT AND REPLAY PASSED: {count} frames, including zero-padded history after resets")
