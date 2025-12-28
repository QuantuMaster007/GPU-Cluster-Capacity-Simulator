from __future__ import annotations

import argparse
import json
import os

from gpu_sim.io import load_yaml, parse_config
from gpu_sim.simulator import run_simulation


def _ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def cmd_run(args: argparse.Namespace) -> int:
    cfg_dict = load_yaml(args.config)
    seed, cluster, pricing, sim, sched, streams = parse_config(cfg_dict)

    result = run_simulation(
        seed=seed,
        cluster_cfg=cluster,
        pricing=pricing,
        sim_cfg=sim,
        sched_cfg=sched,
        streams=streams,
    )

    _ensure_parent_dir(args.out)
    _ensure_parent_dir(args.out_jobs)

    result.tick_df.to_csv(args.out, index=False)
    result.jobs_df.to_csv(args.out_jobs, index=False)

    print("\n=== Simulation Summary ===")
    print(json.dumps(result.summary, indent=2))
    print(f"\nSaved tick metrics: {args.out}")
    print(f"Saved jobs table:   {args.out_jobs}\n")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(prog="gpu-sim", description="GPU cluster capacity simulator.")
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="Run a simulation from a YAML config.")
    run_p.add_argument("--config", required=True, help="Path to YAML config (e.g., configs/base.yaml)")
    run_p.add_argument("--out", default="results/tick_metrics.csv", help="Output CSV for tick metrics")
    run_p.add_argument("--out-jobs", default="results/jobs.csv", help="Output CSV for completed jobs")
    run_p.set_defaults(func=cmd_run)

    args = parser.parse_args()
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
