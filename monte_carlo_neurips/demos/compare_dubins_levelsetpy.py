#!/usr/bin/env python
"""Compare Dubins sampling-based results with levelsetpy.

This script:
1. Computes BRT using the new sampling-based method
2. Computes BRT using levelsetpy
3. Compares 2D slices at different theta values
4. Plots side-by-side comparisons
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, "/home/lex/Documents/ML-Control-Rob/control/levelsetpy")

from utils import SolverConfig
from dubins_python import DubinsSolver
from levelsetpy.grids import createGrid
from levelsetpy.utilities import expand
from levelsetpy.dynamicalsystems import DubinsCar, DRelative
from levelsetpy.initialconditions import cylinder_cost
from levelsetpy.valuefunction import ComputeValueFunctionTTR
from levelsetpy.spatialderivative import computeGradients, upwindFirstENO2
import time


def compute_sampling_brt(theta_vals, mode='simple', grid_res=30, domain=(-2.0, 2.0)):
    """Compute BRT using sampling-based method."""
    print(f"Computing BRT using sampling-based method (mode={mode})...")

    config = SolverConfig(
        delta=0.1,
        num_samples=10_000,
        max_iters=15,
        tol=1e-3,
        t_start=0.0,
        t_end=1.0,
        seed=42,
    )

    solver = DubinsSolver(config=config, mode=mode)

    results = {}
    for theta in theta_vals:
        print(f"  θ = {theta:.2f}...")
        start = time.time()
        X, Z, V, history = solver.solve_slice(
            theta_val=theta,
            grid_res=grid_res,
            domain=domain,
        )
        elapsed = time.time() - start
        results[theta] = {
            'X': X,
            'Z': Z,
            'V': V,
            'history': history,
            'time': elapsed,
        }
        print(f"    Converged in {len(history)} iters, {elapsed:.2f}s")

    return results


def compute_levelsetpy_brt(theta_vals, grid_res=51, domain=(-2.0, 2.0),
                           mode='simple'):
    """Compute BRT using levelsetpy."""
    print(f"Computing BRT using levelsetpy (mode={mode})...")

    # Create grid
    grid_min = expand(np.array([domain[0], domain[0], -np.pi]), ax=1)
    grid_max = expand(np.array([domain[1], domain[1], np.pi]), ax=1)
    N = np.array([[grid_res, grid_res, 25]]).T.astype(np.int64)
    pdDims = [2]  # theta is periodic

    g = createGrid(grid_min, grid_max, N, pdDims)

    # Terminal condition (cylinder)
    radius = 0.5
    data0 = cylinder_cost(g, radius=radius, center=np.zeros(3))

    # Dynamics
    if mode == 'simple':
        # Simple Dubins car: dx/dt = v*cos(θ), dy/dt = v*sin(θ), dθ/dt = ω
        dCar = DubinsCar(
            x=np.zeros((3, 1)),
            wMax=1.0,
            speed=1.0,
            dMax=[0.0],  # No disturbance
        )
    else:
        # Pursuit-evasion (relative Dubins)
        dCar = DRelative(
            x=np.zeros((3, 1)),
            v_p=-1.0,  # Pursuer speed
            v_e=1.0,   # Evader speed
            w=1.0,     # Turn rate
        )

    # Scheme data
    schemeData = {
        'grid': g,
        'dynSys': dCar,
        'accuracy': 'medium',
        'uMode': 'max',
        'dMode': 'min',
    }

    # Time parameters
    t0 = 0.0
    tMax = 1.0
    dt = 0.02
    tau = np.arange(t0, tMax + dt, dt)

    print(f"  Solving HJ PDE backward in time...")
    start = time.time()

    # Compute value function
    try:
        data = ComputeValueFunctionTTR(
            g=g,
            data0=data0,
            tau=tau,
            schemeData=schemeData,
            minWith='zero',
            compRegion=g.xs,
            visualize=False,
        )
        elapsed = time.time() - start
        print(f"  Solved in {elapsed:.2f}s")

        # Extract final time value function
        data_final = data[:, :, :, -1]

    except Exception as e:
        print(f"  Error computing value function: {e}")
        print(f"  Using initial condition instead")
        data_final = data0
        elapsed = time.time() - start

    # Extract slices at different theta values
    results = {}
    for theta in theta_vals:
        # Find closest theta index
        theta_idx = np.argmin(np.abs(g.vs[2] - theta))

        # Extract 2D slice
        V_slice = data_final[:, :, theta_idx]
        X_slice = g.xs[0][:, :, theta_idx]
        Z_slice = g.xs[1][:, :, theta_idx]

        results[theta] = {
            'X': X_slice,
            'Z': Z_slice,
            'V': V_slice,
            'time': elapsed / len(theta_vals),  # Approximate per-slice time
        }
        print(f"  θ = {theta:.2f}, extracted slice")

    return results, g


def plot_comparison(sampling_results, levelsetpy_results, theta_vals, save_path):
    """Plot side-by-side comparison."""
    n_theta = len(theta_vals)
    fig, axes = plt.subplots(n_theta, 3, figsize=(15, 5 * n_theta))

    if n_theta == 1:
        axes = axes.reshape(1, -1)

    for i, theta in enumerate(theta_vals):
        # Sampling method
        X_s = sampling_results[theta]['X']
        Z_s = sampling_results[theta]['Z']
        V_s = sampling_results[theta]['V']

        # levelsetpy
        X_l = levelsetpy_results[theta]['X']
        Z_l = levelsetpy_results[theta]['Z']
        V_l = levelsetpy_results[theta]['V']

        # Plot sampling method
        ax = axes[i, 0]
        cs1 = ax.contourf(X_s, Z_s, V_s, levels=20, cmap='RdBu_r')
        ax.contour(X_s, Z_s, V_s, levels=[0], colors='black', linewidths=2)
        ax.set_title(f'Sampling Method (θ={theta:.2f})')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_aspect('equal')
        plt.colorbar(cs1, ax=ax)

        # Plot levelsetpy
        ax = axes[i, 1]
        cs2 = ax.contourf(X_l, Z_l, V_l, levels=20, cmap='RdBu_r')
        ax.contour(X_l, Z_l, V_l, levels=[0], colors='black', linewidths=2)
        ax.set_title(f'levelsetpy (θ={theta:.2f})')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_aspect('equal')
        plt.colorbar(cs2, ax=ax)

        # Plot difference (interpolate to common grid if needed)
        ax = axes[i, 2]
        if V_s.shape == V_l.shape and np.allclose(X_s, X_l) and np.allclose(Z_s, Z_l):
            diff = V_s - V_l
            cs3 = ax.contourf(X_s, Z_s, diff, levels=20, cmap='seismic')
            ax.set_title(f'Difference (θ={theta:.2f})')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_aspect('equal')
            plt.colorbar(cs3, ax=ax)
        else:
            # Different grid sizes - note this
            ax.text(0.5, 0.5, f'Grid mismatch\nSampling: {V_s.shape}\nlevelsetpy: {V_l.shape}',
                   ha='center', va='center', transform=ax.transAxes, fontsize=10)
            ax.set_title(f'Difference (θ={theta:.2f})')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved comparison plot to {save_path}")
    plt.close()


def compute_metrics(sampling_results, levelsetpy_results, theta_vals):
    """Compute comparison metrics."""
    print("\n" + "="*60)
    print("COMPARISON METRICS")
    print("="*60)

    for theta in theta_vals:
        V_s = sampling_results[theta]['V']
        V_l = levelsetpy_results[theta]['V']

        print(f"\nθ = {theta:.2f}:")
        print(f"  Sampling method:")
        print(f"    Shape: {V_s.shape}")
        print(f"    Min/Max: {V_s.min():.4f} / {V_s.max():.4f}")
        print(f"    Zero-level set area: {np.sum(V_s <= 0)}")
        print(f"    Time: {sampling_results[theta]['time']:.2f}s")
        if 'history' in sampling_results[theta]:
            print(f"    Iterations: {len(sampling_results[theta]['history'])}")

        print(f"  levelsetpy:")
        print(f"    Shape: {V_l.shape}")
        print(f"    Min/Max: {V_l.min():.4f} / {V_l.max():.4f}")
        print(f"    Zero-level set area: {np.sum(V_l <= 0)}")
        print(f"    Time: {levelsetpy_results[theta]['time']:.4f}s")

        # Compute difference metrics if grids match
        if V_s.shape == V_l.shape:
            diff = V_s - V_l
            print(f"  Difference:")
            print(f"    L2 norm: {np.linalg.norm(diff):.4f}")
            print(f"    Max abs: {np.abs(diff).max():.4f}")
            print(f"    Mean abs: {np.abs(diff).mean():.4f}")


def main():
    """Main comparison script."""
    print("="*60)
    print("DUBINS BRT COMPARISON: Sampling vs levelsetpy")
    print("="*60)

    # Parameters
    theta_vals = [0.0, np.pi/4, np.pi/2]
    grid_res_sampling = 30
    grid_res_levelsetpy = 51
    domain = (-2.0, 2.0)
    mode = 'simple'  # or 'pursuit_evasion'

    save_dir = Path(__file__).parent / "results" / "comparisons"
    save_dir.mkdir(parents=True, exist_ok=True)

    # Compute BRTs
    sampling_results = compute_sampling_brt(
        theta_vals, mode=mode, grid_res=grid_res_sampling, domain=domain
    )

    levelsetpy_results, grid = compute_levelsetpy_brt(
        theta_vals, grid_res=grid_res_levelsetpy, domain=domain, mode=mode
    )

    # Plot comparison
    plot_comparison(
        sampling_results,
        levelsetpy_results,
        theta_vals,
        save_path=save_dir / f"dubins_{mode}_comparison.png"
    )

    # Compute metrics
    compute_metrics(sampling_results, levelsetpy_results, theta_vals)

    print("\n" + "="*60)
    print("COMPARISON COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
