#!/usr/bin/env python
"""
1M-bird aerial murmuration safety certification demo.

Multi-predator HJI game with 4D Dubins agents (horizontal position, altitude,
heading).  Demonstrates HJ-Gauss Monte Carlo scaling to 1M agents on GPU with
CPU fallback.

Visual and topological outputs are written to /tmp/murmurations/ at every time
step: trajectory scatter, value heatmap, phase diagram, reachability contours,
and a combined topology-summary panel.  Phase-transition events are appended to
/tmp/murmurations/phase_transitions.txt.

Usage:
    python examples/ex_murmuration.py --device gpu --n-birds 100000
    python examples/ex_murmuration.py --device cpu --n-birds 10000 --save-anim
    python examples/ex_murmuration.py --device gpu --n-birds 1000000 \\
        --n-flocks 10 --n-predators 3 --save-results
    
    # On Lambda 
    python examples/ex_murmuration.py --device gpu --n-birds 1000000 --n-flocks 100 --n-predators 10 --save-results --save-anim --delta 0.05 --max-iters 100 --out-dir /murmurs
"""

import argparse
import os
import time

# Must be set before JAX is imported. JAX's BFC allocator pre-reserves
# MEM_FRACTION × VRAM; the gradient kernel needs 6×(chunk/gpus)×samples×4 B
# live simultaneously (z + g_vals + log_w). At 0.75 (default) on a 16 GB V100
# that cap is 12 GB — too small for large chunks. 0.90 gives 14.4 GB.
# Override at runtime: XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 python ...
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.90")

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import sys
import os
from os.path import join
from pathlib import Path

try:
    _HERE = os.path.abspath(__file__)
except NameError:
    _HERE = str(Path.cwd())
_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, join(_ROOT))

from src.config import SolverConfig
from src.gpu_distribution import GPUDistributor
from src.topology import brt_topology_signature, detect_phase_transitions, TopologyState
from src.output_handler import OutputHandler
from dynamics.murmuration_jax import (
    MurmuationSolverJAX4D,
    FlockState,
    PredatorState,
)


def parse_args():
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description="1M-bird murmuration safety certification on GPU/CPU."
    )
    p.add_argument(
        "--device",
        choices=["cpu", "gpu"],
        default="gpu",
        help="Device: gpu (default) or cpu",
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
        help="MC samples (default 100k)",
    )
    p.add_argument(
        "--max-iters",
        type=int,
        default=100,
        help="Max quasi-linearization iterations (default 100)",
    )
    p.add_argument(
        "--time-steps",
        type=int,
        default=20,
        help="Number of backward time steps for topology tracking (default 20)",
    )
    p.add_argument(
        "--save-results",
        action="store_true",
        help="Save results figure to disk",
    )
    p.add_argument(
        "--save-anim",
        action="store_true",
        help="Save BRT evolution animation (slow)",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default="/tmp/murmurs/results/",
        help="Output directory for summary figures (default results/)",
    )
    p.add_argument(
        "--viz-dir",
        type=str,
        default="/tmp/murmurs",
        help="Directory for per-timestep visualisation outputs",
    )
    p.add_argument(
        "--grid-res",
        type=int,
        default=64,
        help="Resolution of the 2D value-function grid for visualisation (default 64)",
    )
    p.add_argument(
        "--chunk-size",
        type=int,
        default=500_000,
        help="Points per pmap call per GPU pair (default 500k; increase to saturate VRAM)",
    )
    return p.parse_args()


def create_synthetic_flocks(n_birds: int, n_flocks: int, seed: int = 2026) -> list:
    """Create synthetic flock states (4D random initialisation)."""
    key = jax.random.PRNGKey(seed)
    n_per_flock = n_birds // n_flocks

    flocks = []
    for flock_id in range(n_flocks):
        key, subkey = jax.random.split(key)
        states = jax.random.uniform(
            subkey,
            (n_per_flock, 4),
            minval=jnp.array([-5.0, -5.0, 0.0, -jnp.pi]),
            maxval=jnp.array([5.0, 5.0, 100.0, jnp.pi]),
        )
        flocks.append(FlockState(states=states, flock_id=flock_id))

    return flocks


def create_synthetic_predators(n_predators: int, seed: int = 2026) -> list:
    """Create synthetic predator states."""
    key = jax.random.PRNGKey(seed + 1000)
    predators = []
    for _ in range(n_predators):
        key, subkey = jax.random.split(key)
        position = jax.random.uniform(
            subkey,
            (4,),
            minval=jnp.array([-3.0, -3.0, 10.0, -jnp.pi]),
            maxval=jnp.array([3.0, 3.0, 50.0, jnp.pi]),
        )
        predators.append(
            PredatorState(position=position, omega_max=1.0, gamma_max=0.5, speed=1.0)
        )
    return predators


def build_value_grid(
    solver: "MurmuationSolverJAX4D",
    flocks: list,
    predators: list,
    t: float,
    grid_res: int,
    x_extent: tuple = (-5.0, 5.0, -5.0, 5.0),
) -> np.ndarray:
    """Evaluate v on a 2D x1-x2 grid (mean altitude, zero heading).

    Returns shape (grid_res, grid_res).
    """
    x1_min, x1_max, x2_min, x2_max = x_extent
    x1_vals = np.linspace(x1_min, x1_max, grid_res)
    x2_vals = np.linspace(x2_min, x2_max, grid_res)
    X1, X2 = np.meshgrid(x1_vals, x2_vals, indexing="xy")

    # Fix altitude at 50 m, heading at 0
    grid_pts = np.stack(
        [X1.ravel(), X2.ravel(),
         np.full(grid_res * grid_res, 50.0),
         np.zeros(grid_res * grid_res)],
        axis=1,
    ).astype(np.float32)

    grid_jnp = jnp.array(grid_pts)
    # Use first flock / first predator for the 2D grid evaluation
    from src.hamiltonians.murmuration import MurmuationHamiltonian4D
    from dynamics.murmuration_jax import terminal_cost_4d
    from src.hj_sampler import HJReachabilitySampler

    H = MurmuationHamiltonian4D(
        omega_e_bar=solver.omega_e_bar,
        omega_p_bar=solver.omega_p_bar,
        gamma_max=solver.gamma_max,
    )
    hj = HJReachabilitySampler(H, terminal_cost_4d, solver.cfg, solver.distributor)
    v_flat, _ = hj.solve_quasi_linear(grid_jnp, t)
    return np.asarray(v_flat).reshape(grid_res, grid_res)


def main():
    args = parse_args()

    if args.device == "cpu":
        jax.config.update("jax_platform_name", "cpu")

    print("=" * 70)
    print("HJ-Gauss 1M-Bird Aerial Murmuration Safety Certification")
    print("=" * 70)
    print(f"Device: {args.device}")
    print(f"Total birds: {args.n_birds}")
    print(f"Flocks: {args.n_flocks}, Predators: {args.n_predators}")
    print(f"Delta: {args.delta}, Samples: {args.n_samples}, Max iters: {args.max_iters}")
    print(f"Time steps for topology: {args.time_steps}")
    print(f"Visualisation dir: {args.viz_dir}")

    cfg = SolverConfig(
        delta=args.delta,
        num_samples=args.n_samples,
        max_quasi_iters=args.max_iters,
        quasi_tol=1e-5,
        t_start=0.0,
        t_end=2.0,
        gradient_mode="b17",
        chunk_size=args.chunk_size,
        n_flocks=args.n_flocks,
        n_predators=args.n_predators,
        time_steps=args.time_steps,
    )

    print("\nCreating synthetic flocks and predators...")
    flocks = create_synthetic_flocks(args.n_birds, args.n_flocks)
    predators = create_synthetic_predators(args.n_predators)
    print(f"  {len(flocks)} flocks, {sum(f.n_agents for f in flocks)} birds total")
    print(f"  {len(predators)} predators")

    print("\nInitialising GPU distributor...")
    distributor = GPUDistributor(auto_detect=True)

    solver = MurmuationSolverJAX4D(
        cfg=cfg,
        omega_e_bar=1.0,
        omega_p_bar=1.0,
        gamma_max=0.5,
        distributor=distributor,
    )

    # Initialise output handler
    output_handler = OutputHandler(args.viz_dir)

    # Time grid for backward solve
    t_vals = np.linspace(cfg.t_end, cfg.t_start, args.time_steps)
    x_extent = (-5.0, 5.0, -5.0, 5.0)

    # Lists for topology tracking across time steps
    topology_history = []
    chi_list: list = []
    beta1_list: list = []
    n_comp_list: list = []

    print(f"\nRunning {args.time_steps}-step backward solve with per-step visualisation...")
    total_start = time.time()

    # Collect all agent states for trajectory plots
    all_states = np.concatenate([np.asarray(f.states) for f in flocks], axis=0)
    all_flock_ids = np.concatenate(
        [np.full(f.n_agents, f.flock_id) for f in flocks], axis=0
    )

    for step_idx, t_val in enumerate(t_vals):
        t_float = float(t_val)
        print(f"  Step {step_idx:3d}/{args.time_steps}  t={t_float:.3f}", end="  ")
        step_start = time.time()

        # --- Solve BRT at this time step ---
        safety_values, safe_fraction, wall_time = solver.solve_flock_system(
            flocks, predators, t=t_float
        )

        # --- Build 2D value grid for visualisation ---
        v_grid = build_value_grid(
            solver, flocks, predators, t_float,
            grid_res=args.grid_res, x_extent=x_extent,
        )

        # --- Topology ---
        topo = brt_topology_signature(v_grid)
        topo.time_idx = step_idx
        topology_history.append(topo)
        chi_list.append(topo.euler_char)
        beta1_list.append(float(topo.betti_1))
        n_comp_list.append(topo.n_components)

        # --- Detect events at this step ---
        events = detect_phase_transitions(topology_history)
        # Log events from the most recent step only
        for ev_idx, ev_name in events:
            if ev_idx == step_idx:
                output_handler.log_phase_transitions(
                    step_idx, ev_name,
                    topo.euler_char, float(topo.betti_1), topo.n_components
                )

        # --- Per-step visualisations ---
        output_handler.save_trajectory(step_idx, all_states, all_flock_ids)
        output_handler.save_heatmap(step_idx, v_grid, x_extent)
        output_handler.save_phase_diagram(step_idx, all_states, all_flock_ids)
        output_handler.save_reachability(step_idx, v_grid, x_extent)

        # Update topology summary at every step (overwrites previous)
        t_range_so_far = np.arange(step_idx + 1)
        output_handler.save_topology_summary(
            t_range_so_far,
            chi_list,
            beta1_list,
            n_comp_list,
        )

        elapsed = time.time() - step_start
        print(f"safe={safe_fraction:.2%}  chi={topo.euler_char:.2f}  "
              f"beta1={topo.betti_1}  n_comp={topo.n_components}  "
              f"({elapsed:.1f}s)")

    total_wall = time.time() - total_start

    # --- Final single-point solve for overall statistics ---
    print("\nRunning single-point full solve for final statistics...")
    safety_values, safe_fraction, wall_time = solver.solve_flock_system(
        flocks, predators, t=0.0
    )
    all_v_vals = jnp.concatenate(safety_values, axis=0)
    birds_per_sec = args.n_birds / max(wall_time, 1e-6)

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Total wall-clock (all steps): {total_wall:.2f} s")
    print(f"Single-solve wall-clock: {wall_time:.2f} s")
    print(f"Safety fraction (v > 0): {safe_fraction:.2%}")
    print(f"  Safe agents:      {int(safe_fraction * args.n_birds)}/{args.n_birds}")
    print(f"  Threatened agents:{int((1 - safe_fraction) * args.n_birds)}/{args.n_birds}")
    print(f"Throughput: {birds_per_sec:.0f} birds/sec")

    # Phase transition summary
    all_events = detect_phase_transitions(topology_history)
    print(f"\nPhase transitions detected: {len(all_events)}")
    for ev_idx, ev_name in all_events:
        print(f"  t={ev_idx:3d}  {ev_name}")

    print(f"\nOutputs written to: {args.viz_dir}/")
    jpg_count = len(list(Path(args.viz_dir).glob("*.jpg")))
    txt_count = len(list(Path(args.viz_dir).glob("*.txt")))
    print(f"  {jpg_count} .jpg files,  {txt_count} .txt files")

    # 7 IJRR23 swarm action summary
    print("\n7 IJRR23 Swarm Actions - Safety Analysis:")
    print("  1. Flock Cohesion:     formation (r<0.5) outside capture zone")
    print("  2. Heading Consensus:  aligned (|theta|<0.1) at r>0.3")
    print("  3. Predator Evasion:   >=95% outside capture (r>0.2)")
    print("  4. Flash Expansion:    all at r>1.0")
    print("  5. Cordon Formation:   shell at 1.8<r<2.2")
    print("  6. Vacuole Formation:  >=90% outside (r>0.5)")
    print("  7. Voronoi Separation: >=90% per half-plane")

    if args.save_anim:
        from src.topology import animate_brt_evolution
        v_slices = [
            np.load(os.path.join(args.viz_dir, f"v_slice_{i:04d}.npy"))
            for i in range(args.time_steps)
            if os.path.exists(os.path.join(args.viz_dir, f"v_slice_{i:04d}.npy"))
        ]
        if v_slices:
            anim_path = os.path.join(args.viz_dir, "brt_evolution.gif")
            animate_brt_evolution(v_slices, topology_history, all_events, save_path=anim_path)

    if args.save_results:
        os.makedirs(args.out_dir, exist_ok=True)
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(3, 2, figure=fig)

        ax1 = fig.add_subplot(gs[0, :])
        ax1.hist(
            np.asarray(all_v_vals),
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
        for ev_idx, ev_name in all_events:
            ax2.axvline(ev_idx, color="purple", linestyle=":", alpha=0.6)
        ax2.set_xlabel("Time step t")
        ax2.set_ylabel("Topology metric")
        ax2.set_title("BRT topology evolution")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        ax3 = fig.add_subplot(gs[1, 1])
        altitudes = np.array([float(f.states[:, 2].mean()) for f in flocks])
        safety_by_flock = np.array([float(np.mean(np.asarray(sv) > 0)) for sv in safety_values])
        ax3.scatter(altitudes, safety_by_flock, s=100, alpha=0.6, edgecolor="black")
        ax3.set_xlabel("Mean flock altitude (m)")
        ax3.set_ylabel("Safety fraction")
        ax3.set_title("Safety by flock altitude")
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim([-0.05, 1.05])

        ax4 = fig.add_subplot(gs[2, :])
        ax4.axis("off")
        summary_text = (
            "HJ-Gauss Murmuration Safety Certification Summary\n"
            f"  Birds: {args.n_birds} in {args.n_flocks} flocks (4D aerial Dubins)\n"
            f"  Predators: {args.n_predators}  delta={args.delta}  N={args.n_samples}\n"
            f"  Device: {args.device.upper()}\n"
            f"  Wall-clock (single solve): {wall_time:.2f} s\n"
            f"  Throughput: {birds_per_sec:.0f} birds/sec\n"
            f"  Safety: {safe_fraction:.2%} agents outside BRT\n"
            f"  Phase transitions detected: {len(all_events)}"
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
            "1M-Bird Aerial Murmuration Safety Certification",
            fontsize=16, fontweight="bold"
        )
        plt.tight_layout()
        fig_path = os.path.join(args.out_dir, "murmuration_summary.pdf")
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        print(f"\nSaved summary figure to {fig_path}")
        plt.close()

    print("=" * 70)
    print("Demo complete.")


if __name__ == "__main__":
    main()
