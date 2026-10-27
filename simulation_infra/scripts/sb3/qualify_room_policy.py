"""Run the reserved 1,024-trial qualification only after development evaluation meets the target."""

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source/action_space_kit/action_space_kit/rl"))
from qualification import HELD_OUT_SEEDS, audit_reports, validate_challenge


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--reference", default="outputs/full_mission_candidate/params/run.json")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    checkpoint, output = Path(args.checkpoint).resolve(), Path(args.out).resolve()
    manifest = json.loads((checkpoint.parent / "params/run.json").read_text())
    reference = json.loads(Path(args.reference).read_text())
    validate_challenge(manifest, reference)
    match = re.fullmatch(r"model_(\d+)_steps", checkpoint.stem)
    norm = checkpoint.with_name(f"model_vecnormalize_{match[1]}_steps.pkl" if match else "model_vecnormalize.pkl")
    model_hash, norm_hash = (hashlib.sha256(p.read_bytes()).hexdigest() for p in (checkpoint, norm))
    output.mkdir(parents=True, exist_ok=False)
    repo = ROOT.parent
    names = subprocess.check_output(
        ["git", "-C", str(repo), "ls-files", "--cached", "--others", "--exclude-standard"], text=True).splitlines()
    sources = {name: hashlib.sha256((repo / name).read_bytes()).hexdigest() for name in names
               if (repo / name).is_file() and Path(name).suffix in (".py", ".yaml", ".toml")}
    with tarfile.open(output / "evaluation_source.tar.gz", "w:gz") as archive:
        for name in sorted(sources):
            archive.add(repo / name, arcname=name)
    (output / "plan.json").write_text(json.dumps({"checkpoint": str(checkpoint), "checkpoint_sha256": model_hash,
        "normalizer_sha256": norm_hash, "seeds": HELD_OUT_SEEDS, "episodes_per_seed": 256,
        "reference": reference, "candidate": manifest, "evaluation_source_sha256": sources}, indent=2) + "\n")
    env = os.environ.copy()
    for key in ("CONDA_PREFIX", "CONDA_DEFAULT_ENV"):
        env.pop(key, None)
    env.update(OMNI_KIT_ACCEPT_EULA="YES", PYTHONPATH=str(ROOT / "source/action_space_kit"),
               OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    reports = []
    for seed in HELD_OUT_SEEDS:
        report = output / f"seed_{seed}.json"
        argv = [sys.executable, "-u", "scripts/sb3/play.py", "--headless", "--task", manifest["task"],
                "--checkpoint", str(checkpoint), "--num_envs", "256", "--episodes", "256", "--seed", str(seed),
                "--report", str(report), "env.collision_substeps=True"]
        print(f"[qualification] starting seed {seed}", flush=True)
        with (output / f"seed_{seed}.log").open("x") as log:
            result = subprocess.run(argv, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode or not report.is_file():
            raise RuntimeError(f"Evaluation failed; inspect {output / f'seed_{seed}.log'}")
        reports.append(json.loads(report.read_text()))
    summary = audit_reports(reports, manifest["env"], model_hash, norm_hash)
    (output / "qualification.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    print("QUALIFICATION PASSED" if summary["qualification_passed"] else "QUALIFICATION TARGET NOT MET", flush=True)


if __name__ == "__main__":
    main()
