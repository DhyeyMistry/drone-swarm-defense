"""Run independent, pad-stratified headless evaluations and write a promotion report."""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[123, 2026, 907])
    parser.add_argument("--episodes-per-pad", type=int, default=128)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--out", default="outputs/evaluation_suite")
    parser.add_argument("--scenario", choices=("trained", "full", "dropout"), default="full")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    reports = []
    for seed in args.seeds:
        for target in range(4):
            name = f"seed{seed}_pad{target}"
            report = out / (name + ".json")
            cmd = [
                sys.executable,
                "-u",
                str(Path(__file__).with_name("play.py")),
                "--headless",
                "--task",
                "AS-Tello-Room-Single-v0",
                "--checkpoint",
                str(Path(args.checkpoint).resolve()),
                "--num_envs",
                str(args.num_envs),
                "--episodes",
                str(args.episodes_per_pad),
                "--seed",
                str(seed),
                "--report",
                str(report),
                f"env.fixed_spot_index={target}",
            ]
            if args.scenario != "trained":
                cmd += [
                    "env.spawn_box_center=[6.5,0.0]",
                    "env.spawn_box_size=2.0",
                    "env.spawn_range=[6.0,7.0]",
                    "env.takeoff_at_reset=True",
                    "env.episode_length_s=75.0",
                    "env.spawn_near_target_radius=0.0",
                    "env.spawn_short_fraction=0.0",
                    "env.spawn_hover_fraction=0.0",
                ]
            if args.scenario == "dropout":
                cmd += ["env.link_loss_range=[0.15,0.25]"]
            with (out / (name + ".log")).open("w") as handle:
                subprocess.run(cmd, stdout=handle, stderr=subprocess.STDOUT, check=True)
            # Isaac can exit 0 after an exception: a complete JSON report is the pass criterion.
            data = json.loads(report.read_text())
            if not data["summary"]["complete"] or data["summary"]["episodes"] != args.episodes_per_pad:
                raise RuntimeError(f"Incomplete evaluation: {report}")
            reports.append({"seed": seed, "pad": target, "report": str(report), "summary": data["summary"]})
            score = data["summary"]["metrics"]["Episode/success_rate"]
            print(f"{name}: {score:.1%}", flush=True)
    passed = all(
        r["summary"]["metrics"]["Episode/success_rate"] >= 0.95 and r["summary"]["metrics"]["Episode/crash_rate"] == 0
        for r in reports
    )
    (out / "summary.json").write_text(
        json.dumps(
            {
                "scenario": args.scenario,
                "promotion_passed": passed,
                "criteria": (
                    "Every seed and pad >=95% accurate touchdowns, zero crashes; "
                    "inspect Wilson intervals in each report."
                ),
                "results": reports,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"SUITE COMPLETE: promotion_passed={passed}; {out / 'summary.json'}")


if __name__ == "__main__":
    main()
