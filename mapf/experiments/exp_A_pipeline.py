#!/usr/bin/env python
"""Shared windowed geometric planner with alternative execution controllers.

The historical CLI compares geometric execution and first-threat HJ flee.
compare_shields.py also compares simultaneous constraints and a one-threat
ablation using matched dynamics and a new reachability table. The planning
window and BRT horizon are independent parameters. Runtime uses numpy only.
"""
from __future__ import annotations
import os
import sys
import time
import argparse
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(os.path.dirname(_HERE), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from mapf_world import WarehouseGrid, spawn_agents, LifelongTasks
from planner import GeometricPredicate, HJPredicate, plan_window
from executor import DubinsParams, init_cpose, rollout
from hj_conflict import BRTPredicate
from compatible_brt import ReachabilityTable
from compatible_shield import CompatibleShield

DEFAULT_BRT = os.path.join(os.path.dirname(_HERE), "cache", "dubins_brt.npz")

POLICY_GEOMETRIC = "geometric"
POLICY_HJ = "hj"
POLICY_COMPATIBLE = "compatible"
POLICY_SINGLE = "single_constraint"
POLICY_FLEE_MATCHED = "flee_matched"


def make_grid():
    return WarehouseGrid(block_rows=3, block_cols=4, block_h=2, block_w=3,
                         corridor=1, border=2)


def run_episode(seed, policy, *, brt_path=DEFAULT_BRT, margin=0.0,
                n_agents=8, window=6, exec_steps=3, total_ticks=40,
                params: DubinsParams | None = None, grid=None, trace=None,
                barrier_gain=2.0, rate_margin=0.0, fallback="raise"):
    """Run one lifelong-MAPF episode under `policy`. CRN keyed by `seed`.

    Returns dict of headline metrics (all paired-comparable across policies for
    the same seed because starts, goals, priorities and the disturbance stream
    are all functions of `seed` only).
    """
    grid = grid or make_grid()
    params = params or DubinsParams()
    if not 1 <= exec_steps <= window or total_ticks < 1 or n_agents < 1:
        raise ValueError("Require positive ticks/agents and 1 <= exec_steps <= window")
    if n_agents > len(grid.free_cells):
        raise ValueError("More agents than free cells")

    # --- CRN streams (identical across policies for a given seed) -----------
    spawn_rng = np.random.default_rng(seed)
    task_rng = np.random.default_rng(seed + 777)
    dist_rng = np.random.default_rng(seed + 123)

    agents, _ = spawn_agents(grid, n_agents, spawn_rng)
    tasks = LifelongTasks(grid, task_rng)

    # All controllers share the same windowed geometric planner.
    predicate = GeometricPredicate()
    shield = None
    if policy == POLICY_HJ:
        shield = BRTPredicate(brt_path, margin=margin)
    elif policy in (POLICY_COMPATIBLE, POLICY_SINGLE, POLICY_FLEE_MATCHED):
        table = ReachabilityTable(brt_path, margin=margin)
        if policy == POLICY_FLEE_MATCHED:
            shield = table
        else:
            shield = CompatibleShield(table, gain=barrier_gain, rate_margin=rate_margin,
                                      disturbance_bound=params.dist_clip, fallback=fallback,
                                      mode="all" if policy == POLICY_COMPATIBLE else "most_critical")
    elif policy.startswith("chen_mip_"):
        from chen_mip import ChenMIPShield
        table = ReachabilityTable(brt_path, margin=margin)
        shield = ChenMIPShield(table, threshold=float(policy.rsplit("_", 1)[1]),
                              gain=barrier_gain, rate_margin=rate_margin,
                              disturbance_bound=params.dist_clip, fallback=fallback)
    elif policy != POLICY_GEOMETRIC:
        raise ValueError(policy)

    cpose = init_cpose(agents)
    task_start_tick = {a.id: 0 for a in agents}
    goals_reached = 0
    flowtimes = []
    collisions = 0
    wait_steps = 0
    interventions = 0
    plan_time = 0.0
    macro_steps = 0
    coll_series = []   # collisions per tick (for MSER-5 warm-up truncation)
    goal_series = []   # goals per tick
    shield_diagnostics = {}

    for tick in range(total_ticks):
        t0 = time.time()
        plans, pstats = plan_window(grid, agents, predicate, window)
        plan_time += time.time() - t0
        wait_steps += pstats["wait_steps"]

        info = rollout(grid, agents, plans, cpose, params, dist_rng,
                       exec_steps, shield=shield, trace=trace)
        collisions += info["collisions"]
        interventions += info["interventions"]
        for key, value in info.items():
            if key not in ("collisions", "interventions"):
                shield_diagnostics[key] = shield_diagnostics.get(key, 0) + value
        macro_steps += exec_steps
        coll_series.append(float(info["collisions"]))

        # lifelong goal reassignment
        goals_this_tick = 0
        for a in agents:
            if a.cell == a.goal:
                a.goals_reached += 1
                goals_reached += 1
                goals_this_tick += 1
                flowtimes.append((tick + 1) - task_start_tick[a.id])
                task_start_tick[a.id] = tick + 1
                a.goal = tasks.new_goal(exclude=a.cell)
        goal_series.append(float(goals_this_tick))

    throughput = goals_reached / max(macro_steps, 1)
    return {
        "seed": int(seed),
        "policy": policy,
        "throughput": float(throughput),
        "collisions": float(collisions),
        "goals_reached": float(goals_reached),
        "mean_flowtime": float(np.mean(flowtimes)) if flowtimes else float("nan"),
        "wait_steps": float(wait_steps),
        "interventions": float(interventions),
        "plan_time_s": float(plan_time),
        "macro_steps": int(macro_steps),
        "n_agents": int(n_agents),
        "window": int(window),
        "coll_series": coll_series,
        "goal_series": goal_series,
        "disturbance_mode": params.disturbance_mode,
        "simulated_seconds": macro_steps * params.macro_T,
        "replanning_period_seconds": exec_steps * params.macro_T,
        "planning_window_seconds": window * params.macro_T,
        "task_progress": "committed_discrete_cells",
        "sampled_safety_certified": False,
        **shield_diagnostics,
    }


def _fmt(d):
    return (f"  {d['policy']:<10}  thru={d['throughput']:.4f}  "
            f"collisions={d['collisions']:.0f}  goals={d['goals_reached']:.0f}  "
            f"flowtime={d['mean_flowtime']:.2f}  waits={d['wait_steps']:.0f}  "
            f"shield={d.get('interventions', 0):.0f}  plan={d['plan_time_s']:.2f}s")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-agents", type=int, default=8)
    ap.add_argument("--window", type=int, default=6)
    ap.add_argument("--exec-steps", type=int, default=3)
    ap.add_argument("--total-ticks", type=int, default=40)
    ap.add_argument("--margin", type=float, default=0.0)
    ap.add_argument("--brt", type=str, default=DEFAULT_BRT)
    args = ap.parse_args()

    if not os.path.exists(args.brt):
        print(f"[FATAL] BRT cache not found: {args.brt}\n"
              f"        Run:  ./.venv/bin/python experiments/precompute_brt.py")
        sys.exit(2)

    grid = make_grid()
    print(f"Grid {grid.H}x{grid.W}  free={len(grid.free_cells)}  "
          f"endpoints={len(grid.endpoints)}  agents={args.n_agents}  "
          f"window={args.window}  seed={args.seed}")
    kw = dict(brt_path=args.brt, margin=args.margin, n_agents=args.n_agents,
              window=args.window, exec_steps=args.exec_steps,
              total_ticks=args.total_ticks, grid=grid)
    print("\nInitial trial (single seed, CRN paired):")
    g = run_episode(args.seed, POLICY_GEOMETRIC, **kw)
    h = run_episode(args.seed, POLICY_HJ, **kw)
    print(_fmt(g))
    print(_fmt(h))
    dc = g["collisions"] - h["collisions"]
    dt = (h["throughput"] - g["throughput"]) / max(g["throughput"], 1e-9) * 100
    print(f"\n  Δcollisions (geom - hj) = {dc:.0f}   "
          f"throughput change (hj vs geom) = {dt:+.1f}%")


if __name__ == "__main__":
    main()
