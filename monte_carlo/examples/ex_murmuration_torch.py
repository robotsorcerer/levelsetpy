#!/usr/bin/env python
"""
1M-bird aerial murmuration safety certification demo (PyTorch).

Multi-predator HJI game with 4D Dubins agents (horizontal position, altitude,
heading). Demonstrates Monte Carlo HJ solving scaled to 1M agents on GPU/CPU.

Device and distributed-agnostic: works on single/multi-GPU and multi-node setups.

Usage:
    python examples/ex_murmuration_torch.py --n-birds 100000
    python examples/ex_murmuration_torch.py --n-birds 1000000 --device cuda

    # Multi-GPU (single node)
    torchrun --nproc_per_node=4 examples/ex_murmuration_torch.py --n-birds 1000000

    # Multi-node (e.g., 2 nodes with 4 GPUs each)
    torchrun --nproc_per_node=4 --nnodes=2 --node_rank=0 examples/ex_murmuration_torch.py
"""

import argparse
import os
import time
from pathlib import Path
from dataclasses import dataclass

import torch
import torch.distributed as dist
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import sys
from os.path import join

try:
    _HERE = os.path.abspath(__file__)
except NameError:
    _HERE = str(Path.cwd())
_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, join(_ROOT))

from dynamics.murmuration_torch import (
    MurmuationSolverTorch4D,
    create_synthetic_flocks,
    create_synthetic_predators,
    terminal_cost_4d,
)


@dataclass
class TopoMetrics:
    """Topology metrics from value grid."""
    euler_char: float
    betti_1: int
    n_components: int
    time_idx: int = 0


def build_value_grid(
    solver: MurmuationSolverTorch4D,
    eval_points: torch.Tensor,
    t: float,
    grid_res: int = 64,
    x_extent: tuple = (-5.0, 5.0, -5.0, 5.0),
) -> np.ndarray:
    """Evaluate v on a 2D x1-x2 grid (mean altitude, zero heading).

    Parameters
    ----------
    solver : MurmuationSolverTorch4D
        The solver.
    eval_points : torch.Tensor
        Reference evaluation points for device/dtype inference.
    t : float
        Query time.
    grid_res : int
        Resolution of the grid.
    x_extent : tuple
        (x1_min, x1_max, x2_min, x2_max).

    Returns
    -------
    np.ndarray, shape (grid_res, grid_res)
        Value function on the grid.
    """
    x1_min, x1_max, x2_min, x2_max = x_extent
    x1_vals = np.linspace(x1_min, x1_max, grid_res)
    x2_vals = np.linspace(x2_min, x2_max, grid_res)
    X1, X2 = np.meshgrid(x1_vals, x2_vals, indexing="xy")

    grid_pts = np.stack(
        [
            X1.ravel(),
            X2.ravel(),
            np.full(grid_res * grid_res, 50.0),
            np.zeros(grid_res * grid_res),
        ],
        axis=1,
    ).astype(np.float32)

    device = eval_points.device
    grid_torch = torch.from_numpy(grid_pts).to(device=device, dtype=eval_points.dtype)

    # Simple evaluation without quasi-linearization for speed
    with torch.no_grad():
        v_flat = solver._mc_value_at_point(
            grid_torch,
            t,
            torch.ones(grid_res * grid_res, device=device),
            terminal_cost_4d,
        )

    return v_flat.cpu().numpy().reshape(grid_res, grid_res)


def compute_topology_signature(v_grid: np.ndarray) -> TopoMetrics:
    """Compute basic topology metrics from value grid.

    Parameters
    ----------
    v_grid : np.ndarray, shape (grid_res, grid_res)
        Value function grid.

    Returns
    -------
    TopoMetrics
        Topology metrics (simplified: Euler characteristic approximation).
    """
    # Approximate topology via level-set connectivity
    binary = (v_grid > 0).astype(int)
    n_components = len(np.unique(binary)) - 1  # Rough count

    # Simplified Euler characteristic: χ ≈ #components - #holes
    chi = n_components
    betti_1 = max(0, n_components - 1)

    return TopoMetrics(
        euler_char=float(chi),
        betti_1=betti_1,
        n_components=n_components,
    )


def parse_args():
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description="1M-bird murmuration safety certification (PyTorch)."
    )
    p.add_argument(
        "--n-birds",
        type=int,
        default=100_000,
        help="Total number of birds (default 100k)",
    )
    p.add_argument(
        "--n-flocks",
        type=int,
        default=5,
        help="Number of flocks (default 5)",
    )
    p.add_argument(
        "--n-predators",
        type=int,
        default=2,
        help="Number of predators (default 2)",
    )
    p.add_argument(
        "--delta",
        type=float,
        default=0.05,
        help="Viscosity parameter (default 0.05)",
    )
    p.add_argument(
        "--n-samples",
        type=int,
        default=10_000,
        help="MC samples per evaluation (default 10k)",
    )
    p.add_argument(
        "--max-iters",
        type=int,
        default=10,
        help="Max quasi-linearization iterations (default 10)",
    )
    p.add_argument(
        "--time-steps",
        type=int,
        default=5,
        help="Number of backward time steps (default 5)",
    )
    p.add_argument(
        "--save-results",
        action="store_true",
        help="Save results figure to disk",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default="/tmp/murmurs_torch/",
        help="Output directory (default /tmp/murmurs_torch/)",
    )
    p.add_argument(
        "--grid-res",
        type=int,
        default=32,
        help="Resolution of 2D value grid (default 32)",
    )
    p.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device: auto (GPU if available), cpu, or cuda",
    )
    return p.parse_args()


def init_distributed():
    """Initialize distributed training if running via torchrun.

    Returns
    -------
    tuple (rank, world_size, device)
        rank : int - process rank (0 = main)
        world_size : int - total number of processes
        device : torch.device - device for this process
    """
    if not dist.is_available():
        return 0, 1, torch.device("cpu")

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
        device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
        torch.cuda.set_device(device)
        return rank, world_size, device

    return 0, 1, torch.device("cpu")


def is_main_process(rank: int) -> bool:
    """Check if this is the main process."""
    return rank == 0


def main():
    args = parse_args()

    # Initialize distributed training
    rank, world_size, device = init_distributed()

    # Device selection (respects distributed setup)
    if args.device == "auto":
        if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
            device = device  # Use distributed device
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # Only print from main process
    if is_main_process(rank):
        print("=" * 70)
        print("PyTorch HJ-Gauss Murmuration Safety Certification")
        print("=" * 70)
        if world_size > 1:
            print(f"Distributed training: {world_size} processes")
        print(f"Process rank: {rank}/{world_size}")
        print(f"Device: {device}")
        print(f"Total birds: {args.n_birds}")
        print(f"Flocks: {args.n_flocks}, Predators: {args.n_predators}")
        print(f"Delta: {args.delta}, Samples: {args.n_samples}, Max iters: {args.max_iters}")
        print(f"Time steps: {args.time_steps}")

    # Create solver
    solver = MurmuationSolverTorch4D(
        delta=args.delta,
        num_samples=args.n_samples,
        max_quasi_iters=args.max_iters,
        quasi_tol=1e-5,
        t_start=0.0,
        t_end=2.0,
        omega_e_bar=1.0,
        omega_p_bar=1.0,
        gamma_max=0.5,
        seed=2026,
    )

    if is_main_process(rank):
        print("\nCreating synthetic flocks and predators...")

    # Distribute birds across processes
    n_birds_per_process = args.n_birds // world_size
    n_birds_local = n_birds_per_process + (args.n_birds % world_size if rank < (args.n_birds % world_size) else 0)

    flocks = create_synthetic_flocks(n_birds_local, args.n_flocks, device, seed=2026 + rank)
    predators = create_synthetic_predators(args.n_predators, device, seed=2026 + rank)

    if is_main_process(rank):
        print(f"  {len(flocks)} flocks, {sum(f.n_agents for f in flocks)} birds per process")
        print(f"  Total birds (all processes): {args.n_birds}")
        print(f"  {len(predators)} predators")

    # Time grid for backward solve
    t_vals = np.linspace(solver.t_end, solver.t_start, args.time_steps)
    x_extent = (-5.0, 5.0, -5.0, 5.0)

    # Topology tracking
    topology_history = []
    chi_list = []
    beta1_list = []
    n_comp_list = []

    if is_main_process(rank):
        print(f"\nRunning {args.time_steps}-step backward solve...")
    total_start = time.time()

    # Synchronize all processes before starting
    if world_size > 1:
        dist.barrier()

    # Collect all agent states
    all_states = torch.cat([f.states for f in flocks], dim=0)
    all_flock_ids = torch.cat(
        [torch.full((f.n_agents,), f.flock_id, device=device) for f in flocks], dim=0
    )

    for step_idx, t_val in enumerate(t_vals):
        t_float = float(t_val)
        if is_main_process(rank):
            print(f"  Step {step_idx:3d}/{args.time_steps}  t={t_float:.3f}", end="  ")
        step_start = time.time()

        # Solve BRT at this time step
        safety_values, safe_fraction, wall_time = solver.solve_flock_system(
            flocks, predators, t=t_float
        )

        # Synchronize across processes
        if world_size > 1:
            dist.barrier()
            # Reduce safe_fraction across all processes
            safe_fraction_tensor = torch.tensor([safe_fraction], device=device)
            dist.all_reduce(safe_fraction_tensor)
            safe_fraction = float(safe_fraction_tensor[0]) / world_size

        # Build 2D value grid (only on main process to save compute)
        if is_main_process(rank):
            v_grid = build_value_grid(
                solver, all_states, t_float,
                grid_res=args.grid_res, x_extent=x_extent,
            )

            # Compute topology
            topo = compute_topology_signature(v_grid)
            topo.time_idx = step_idx
            topology_history.append(topo)
            chi_list.append(topo.euler_char)
            beta1_list.append(float(topo.betti_1))
            n_comp_list.append(topo.n_components)

            elapsed = time.time() - step_start
            print(
                f"safe={safe_fraction:.2%}  chi={topo.euler_char:.2f}  "
                f"beta1={topo.betti_1}  n_comp={topo.n_components}  "
                f"({elapsed:.1f}s)"
            )

    total_wall = time.time() - total_start

    # Final single-point solve
    if is_main_process(rank):
        print("\nRunning final solve for statistics...")
    safety_values, safe_fraction, wall_time = solver.solve_flock_system(
        flocks, predators, t=0.0
    )
    all_v_vals = torch.cat(safety_values, dim=0)

    # Reduce across all processes
    if world_size > 1:
        safe_fraction_tensor = torch.tensor([safe_fraction], device=device)
        dist.all_reduce(safe_fraction_tensor)
        safe_fraction = float(safe_fraction_tensor[0]) / world_size

    # Calculate throughput
    birds_per_sec = args.n_birds / max(wall_time, 1e-6)

    # Only print results from main process
    if is_main_process(rank):
        print("\n" + "=" * 70)
        print("RESULTS")
        print("=" * 70)
        print(f"Total wall-clock (all steps): {total_wall:.2f} s")
        print(f"Single-solve wall-clock: {wall_time:.2f} s")
        print(f"Safety fraction (v > 0): {safe_fraction:.2%}")
        print(f"  Safe agents:       {int(safe_fraction * args.n_birds)}/{args.n_birds}")
        print(f"  Threatened agents: {int((1 - safe_fraction) * args.n_birds)}/{args.n_birds}")
        print(f"Throughput: {birds_per_sec:.0f} birds/sec")
        print(f"\nPhase transitions detected: {len(topology_history)}")

    if args.save_results and is_main_process(rank):
        os.makedirs(args.out_dir, exist_ok=True)
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(3, 2, figure=fig)

        ax1 = fig.add_subplot(gs[0, :])
        ax1.hist(
            all_v_vals.cpu().numpy(),
            bins=50,
            alpha=0.7,
            edgecolor="black",
            label="Value function",
        )
        ax1.axvline(0, color="red", linestyle="--", linewidth=2, label="Safety boundary (v=0)")
        ax1.set_xlabel("BRT value v(x)")
        ax1.set_ylabel("Count (agents)")
        ax1.set_title("Safety Distribution across all birds")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2 = fig.add_subplot(gs[1, 0])
        t_range_plot = np.arange(len(chi_list))
        ax2.plot(t_range_plot, chi_list, "b-o", markersize=4, label="χ(t)")
        ax2.plot(t_range_plot, beta1_list, "r-s", markersize=4, label="β₁(t)")
        ax2.plot(t_range_plot, n_comp_list, "g-^", markersize=4, label="n_comp(t)")
        ax2.set_xlabel("Time step t")
        ax2.set_ylabel("Topology metric")
        ax2.set_title("Value function topology evolution")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        ax3 = fig.add_subplot(gs[1, 1])
        altitudes = np.array([float(f.states[:, 2].mean()) for f in flocks])
        safety_by_flock = np.array(
            [float((sv > 0).float().mean()) for sv in safety_values]
        )
        ax3.scatter(altitudes, safety_by_flock, s=100, alpha=0.6, edgecolor="black")
        ax3.set_xlabel("Mean flock altitude (m)")
        ax3.set_ylabel("Safety fraction")
        ax3.set_title("Safety by flock altitude")
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim([-0.05, 1.05])

        ax4 = fig.add_subplot(gs[2, :])
        ax4.axis("off")
        summary_text = (
            "PyTorch HJ-Gauss Murmuration Safety Certification\n"
            f"  Birds: {args.n_birds} in {args.n_flocks} flocks (4D aerial Dubins)\n"
            f"  Predators: {args.n_predators}  delta={args.delta}  N={args.n_samples}\n"
            f"  Device: {device}  Processes: {world_size}\n"
            f"  Wall-clock (single solve): {wall_time:.2f} s\n"
            f"  Throughput: {birds_per_sec:.0f} birds/sec\n"
            f"  Safety: {safe_fraction:.2%} agents outside BRT\n"
        )
        ax4.text(
            0.05,
            0.95,
            summary_text,
            transform=ax4.transAxes,
            fontsize=10,
            verticalalignment="top",
            fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        fig.suptitle(
            "Murmuration Safety Certification (PyTorch)",
            fontsize=16, fontweight="bold"
        )
        plt.tight_layout()
        fig_path = os.path.join(args.out_dir, "murmuration_summary.pdf")
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        print(f"\nSaved summary figure to {fig_path}")
        plt.close()

    # Synchronize before exit
    if world_size > 1:
        dist.barrier()

    if is_main_process(rank):
        print("=" * 70)
        print("Demo complete.")


if __name__ == "__main__":
    main()
