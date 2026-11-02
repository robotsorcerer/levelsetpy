#!/usr/bin/env python3
"""Paired execution-controller comparison with matched table and dynamics.

The one-constraint arm is an ablation, not a reproduction of Chen's MIP.
Research runs explicitly permit least-violation control and report failures.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from exp_A_pipeline import run_episode
from executor import DubinsParams
from compatible_brt import ReachabilityTable

ROOT = Path(__file__).resolve().parents[1]
POLICIES = ("geometric", "flee_matched", "single_constraint", "compatible")


def paired_difference(records, baseline, *, draws=10000):
    ours = {row["seed"]: row["collisions"] for row in records if row["policy"] == "compatible"}
    base = {row["seed"]: row["collisions"] for row in records if row["policy"] == baseline}
    delta = np.array([ours[seed] - base[seed] for seed in sorted(ours)])
    samples = np.random.default_rng(913).choice(delta, (draws, len(delta)), replace=True).mean(axis=1)
    return dict(mean=float(delta.mean()), percentile_bootstrap_95=np.quantile(samples, [.025, .975]).tolist(),
                difference="compatible minus " + baseline, paired_seeds=len(delta))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--brt", type=Path, default=ROOT / "cache/compatible_brt.npz")
    parser.add_argument("--out", type=Path, default=ROOT / "results/compatible_comparison.json")
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--policies", nargs="+", default=list(POLICIES))
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--n-agents", type=int, default=14)
    parser.add_argument("--total-ticks", type=int, default=40)
    parser.add_argument("--window", type=int, default=6)
    parser.add_argument("--exec-steps", type=int, default=3)
    parser.add_argument("--substeps", type=int, default=10)
    parser.add_argument("--margin", type=float, default=.1)
    parser.add_argument("--barrier-gain", type=float, default=2.)
    parser.add_argument("--rate-margin", type=float, default=0.)
    parser.add_argument("--dist-sigma", type=float, default=.05)
    parser.add_argument("--dist-clip", type=float, default=.12)
    args = parser.parse_args()
    if args.seeds < 1 or args.start_seed < 0:
        parser.error("Require a positive seed count and a nonnegative starting seed")
    table = ReachabilityTable(args.brt, margin=args.margin)
    params = DubinsParams(substeps=args.substeps, dist_sigma=args.dist_sigma,
                          dist_clip=args.dist_clip, disturbance_mode="velocity")
    records = []
    for seed in range(args.start_seed, args.start_seed + args.seeds):
        for policy in args.policies:
            result = run_episode(seed, policy, brt_path=args.brt, margin=args.margin,
                                 n_agents=args.n_agents, total_ticks=args.total_ticks,
                                 window=args.window, exec_steps=args.exec_steps, params=params,
                                 barrier_gain=args.barrier_gain, rate_margin=args.rate_margin,
                                 fallback="least_violation")
            # Physical task completion is not implemented; do not report grid
            # progress as a measured execution-controller throughput result.
            for key in ("throughput", "goals_reached", "mean_flowtime", "goal_series", "wait_steps"):
                result.pop(key)
            records.append(result)
            print(f"seed={seed:2d} {policy:17s} contacts={result['collisions']:3.0f} "
                  f"violated={result['unsatisfied_agent_steps']:5d} "
                  f"multi-compatible={result['compatible_multi_threat_steps']:5d} "
                  f"shield={result['shield_time_s']:.2f}s", flush=True)
    summaries = {}
    keys = ("collisions", "infeasible_agent_steps", "unsatisfied_agent_steps", "replan_agent_steps",
            "multi_threat_agent_steps", "compatible_multi_threat_steps", "shield_time_s",
            "negative_barrier_agent_steps", "outside_domain_agent_steps")
    for policy in args.policies:
        selected = [row for row in records if row["policy"] == policy]
        summaries[policy] = {"mean_" + key: float(np.mean([r[key] for r in selected])) for key in keys}
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config["brt"] = args.brt.name
    config["out"] = args.out.name
    output = dict(config=config, table_metadata=table.meta,
                  table_sha256=hashlib.sha256(args.brt.read_bytes()).hexdigest(),
                  geometric_padding=table.geometric_padding,
                  disturbance_mode="bounded_world_velocity", dt=params.dt,
                  simulated_seconds_per_episode=args.total_ticks * args.exec_steps * params.macro_T,
                  agent_steps_per_episode=args.total_ticks * args.exec_steps * args.substeps * args.n_agents,
                  collision_metric="new contact onsets over swept Euler segments; initial contact excluded",
                  safety_certified=False, fallback="least_violation",
                  limitations=["Chen MIP arm uses a matched finite-horizon table, not its infinite-horizon certificate", "No physical task-completion measurement",
                               "No obstacle-avoidance constraint", "No recursive-feasibility or sampled-data certificate",
                               "Non-barrier baselines do not populate constraint diagnostics"],
                  summaries=summaries,
                  paired_collision_differences={policy: paired_difference(records, policy) for policy in args.policies if policy != "compatible"} if "compatible" in args.policies else {},
                  records=records)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    print(json.dumps(output["paired_collision_differences"], indent=2))
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
