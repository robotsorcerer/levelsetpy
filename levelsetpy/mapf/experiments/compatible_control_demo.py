#!/usr/bin/env python3
"""Reproducible three-robot compatibility example and infeasibility example.

The poses were selected to expose a constraint missed by a one-threat filter.
This is a mechanism demonstration, not an unbiased performance benchmark.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from compatible_brt import ReachabilityTable
from compatible_shield import CompatibleShield, project_interval
from hj_conflict import wrap_angle

INITIAL_POSES = np.array([[0., 0., 0.],
                         [1.9604650670584451, .9734315411078986, -1.1574429188830908],
                         [.6774581645330908, 1.4318998032889785, -2.5164186378709923]])


def demonstrate(table, *, dt=.01, steps=300):
    results = {}
    for mode in ("all", "most_critical"):
        shield = CompatibleShield(table, fallback="least_violation", mode=mode)
        poses = INITIAL_POSES.copy()
        initial = shield.filter(poses, np.ones(3), np.ones(3), dt=dt)
        frames, violations, compatible_multi = [], 0, 0
        for step in range(steps):
            decision = shield.filter(poses, np.ones(3), np.ones(3), dt=dt)
            violations += int(np.count_nonzero(~decision.all_constraints_satisfied))
            compatible_multi += int(np.count_nonzero((decision.active_pairs >= 2)
                                                     & decision.all_constraints_satisfied))
            previous = poses.copy()
            poses[:, :2] += dt * np.column_stack((np.cos(poses[:, 2]), np.sin(poses[:, 2])))
            poses[:, 2] = wrap_angle(poses[:, 2] + dt * decision.turns)
            i, j = np.triu_indices(3, 1)
            start = previous[i, :2] - previous[j, :2]
            change = poses[i, :2] - poses[j, :2] - start
            denom = np.sum(change ** 2, axis=1)
            fraction = np.clip(np.divide(-np.sum(start * change, axis=1), denom,
                                         out=np.zeros_like(denom), where=denom > 0), 0, 1)
            separation = np.linalg.norm(start + fraction[:, None] * change, axis=1)
            frames.append(dict(time=(step + 1) * dt, poses=poses.tolist(),
                               min_separation=float(separation.min()),
                               min_barrier=float(decision.min_barrier.min()),
                               min_residual=float(decision.min_residual.min())))
        results[mode] = dict(initial_turns=initial.turns.tolist(),
                             initial_intervals=np.column_stack((initial.lower, initial.upper)).tolist(),
                             initial_min_residual=initial.min_residual.tolist(),
                             initial_min_barrier=initial.min_barrier.tolist(),
                             initial_active_pairs=initial.active_pairs.tolist(),
                             violated_agent_steps=violations,
                             compatible_multi_threat_steps=compatible_multi,
                             min_separation=min(f["min_separation"] for f in frames),
                             min_barrier=min(f["min_barrier"] for f in frames), frames=frames)
    impossible = project_interval(1., [1., -1.], [-.7, -.7], 1.)
    return dict(table_metadata=table.meta, initial_poses=INITIAL_POSES.tolist(),
                margin=table.margin, dt=dt, steps=steps, nominal_turns=[1., 1., 1.],
                controller_disturbance_bound=.12, realized_disturbance=0.,
                safety_certified=False, scenario_selection="selected mechanism demonstration",
                modes=results,
                incompatible_example=dict(constraints=["omega >= 0.7", "omega <= -0.7"],
                                          feasible=impossible.feasible, fallback_turn=impossible.control,
                                          minimum_residual=impossible.min_residual))


def plot_demo(data, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 3, figsize=(12, 3.5), layout="constrained")
    for mode, style, label in [("all", "-", "All constraints"),
                                ("most_critical", "--", "One constraint")]:
        frames = data["modes"][mode]["frames"]
        poses = np.array([data["initial_poses"]] + [f["poses"] for f in frames])
        for robot in range(3):
            axes[0].plot(poses[:, robot, 0], poses[:, robot, 1], style,
                         color=f"C{robot}", label=f"Robot {robot}: {label}")
            if mode == "all":
                axes[0].annotate(str(robot), poses[0, robot, :2], xytext=(3, 4),
                                 textcoords="offset points", color=f"C{robot}")
        times = [f["time"] for f in frames]
        axes[1].plot(times, [f["min_separation"] for f in frames], style, label=label)
        early = [f for f in frames if f["time"] <= .45]
        axes[2].plot([f["time"] for f in early], [f["min_residual"] for f in early], style, label=label)
    axes[0].set(xlabel="x (m)", ylabel="y (m)", title="Three-robot trajectories", aspect="equal")
    axes[1].axhline(data["table_metadata"]["capture_radius"], color="black", lw=.8, label="Contact distance")
    axes[1].set(xlabel="Time (s)", ylabel="Separation (m)", title="Minimum pairwise separation")
    axes[2].axhline(0, color="black", lw=.8)
    axes[2].set(xlabel="Time (s)", ylabel="Minimum constraint residual", title="Early all-pair residuals")
    axes[1].legend(fontsize=8)
    axes[2].legend(fontsize=8)
    for axis in axes:
        axis.grid(alpha=.2)
    figure.savefig(path)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--brt", type=Path, default=ROOT / "cache/compatible_brt.npz")
    parser.add_argument("--out", type=Path, default=ROOT / "results/compatible_three_robot.json")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    table = ReachabilityTable(args.brt, margin=.1)
    result = demonstrate(table)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    for mode, metrics in result["modes"].items():
        print(mode, json.dumps({k: v for k, v in metrics.items() if k != "frames"}, indent=2))
    if args.plot:
        plot_demo(result, args.out.with_suffix(".svg"))
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
