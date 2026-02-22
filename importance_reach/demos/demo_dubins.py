#!/usr/bin/env python
"""Demo script for Dubins car BRT computation using sampling method.

Visualizes:
1. Value function at different theta slices
2. Zero-level sets (BRT boundary)
3. Convergence history
4. Comparison between simple and pursuit-evasion modes
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D

sys.path.insert(0, str(Path(__file__).parent))

from src.sampling_engine import SolverConfig
from dynamics.dubins_python import DubinsSolver


def run_mode(mode, theta_vals, grid_res, domain):
    """Run solver for a specific mode."""
    print(f"\n{'='*60}")
    print(f"MODE: {mode.upper()}")
    print('='*60)

    config = SolverConfig(
        delta=0.1,
        num_samples=10_000,
        max_iters=20,
        tol=1e-3,
        t_start=0.0,
        t_end=1.0,
        seed=42,
    )

    solver = DubinsSolver(config=config, mode=mode)

    results = []
    for theta in theta_vals:
        print(f"\nComputing BRT at θ = {theta:.3f} ({np.degrees(theta):.1f}°)...")
        X, Z, V, history = solver.solve_slice(
            theta_val=theta,
            grid_res=grid_res,
            domain=domain,
        )
        results.append({
            'theta': theta,
            'X': X,
            'Z': Z,
            'V': V,
            'history': history,
        })
        print(f"  Converged in {len(history)} iterations")
        print(f"  Final residual: {history[-1]:.6f}")
        print(f"  Value range: [{V.min():.4f}, {V.max():.4f}]")
        print(f"  Zero-level set points: {np.sum(V <= 0)}")

    return results


def plot_results(results, mode, save_dir):
    """Plot results for a specific mode."""

    # Plot 1: Value functions at different theta
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i, result in enumerate(results[:6]):
        ax = axes[i]
        X = result['X']
        Z = result['Z']
        V = result['V']
        theta = result['theta']

        # Contour plot
        cs = ax.contourf(X, Z, V, levels=20, cmap='RdBu_r', vmin=-1, vmax=1)
        ax.contour(X, Z, V, levels=[0], colors='black', linewidths=2)

        # Add target circle
        circle = plt.Circle((0, 0), 0.5, fill=False, color='green',
                          linestyle='--', linewidth=2, label='Target')
        ax.add_patch(circle)

        ax.set_title(f'θ = {np.degrees(theta):.1f}°')
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        plt.colorbar(cs, ax=ax, label='Value')

    fig.suptitle(f'Dubins Car BRT - {mode.replace("_", " ").title()} Mode',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_dir / f"dubins_{mode}_slices.png", dpi=150, bbox_inches='tight')
    print(f"Saved: {save_dir / f'dubins_{mode}_slices.png'}")
    plt.close()

    # Plot 2: Convergence history and zero-level sets
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    for result in results:
        theta = result['theta']
        history = result['history']
        ax.semilogy(history, label=f'θ={np.degrees(theta):.1f}°', marker='o')

    ax.set_xlabel('Iteration')
    ax.set_ylabel('Relative Residual')
    ax.set_title(f'Convergence History - {mode.replace("_", " ").title()}')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Zero-level sets overlay
    ax = axes[1]
    for result in results:
        theta = result['theta']
        X = result['X']
        Z = result['Z']
        V = result['V']

        ax.contour(X, Z, V, levels=[0], linewidths=2,
                  label=f'θ={np.degrees(theta):.1f}°')

    # Add target
    circle = plt.Circle((0, 0), 0.5, fill=False, color='green',
                       linestyle='--', linewidth=2, label='Target')
    ax.add_patch(circle)

    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_title(f'BRT Boundaries - {mode.replace("_", " ").title()}')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=8)

    plt.tight_layout()
    plt.savefig(save_dir / f"dubins_{mode}_convergence.png", dpi=150, bbox_inches='tight')
    print(f"Saved: {save_dir / f'dubins_{mode}_convergence.png'}")
    plt.close()

    # Plot 3: 3D visualization
    result = results[2]  # Use theta = pi/4
    X = result['X']
    Z = result['Z']
    V = result['V']
    theta = result['theta']

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(X, Z, V, cmap='RdBu_r', alpha=0.8,
                          vmin=-1, vmax=1, edgecolor='none')

    # Add zero-level contour
    ax.contour(X, Z, V, levels=[0], colors='black', linewidths=3,
              offset=V.min())

    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_zlabel('Value')
    ax.set_title(f'{mode.replace("_", " ").title()} - Value Function (θ = {np.degrees(theta):.1f}°)')
    fig.colorbar(surf, ax=ax, shrink=0.5, label='Value')

    plt.savefig(save_dir / f"dubins_{mode}_3d.png", dpi=150, bbox_inches='tight')
    print(f"Saved: {save_dir / f'dubins_{mode}_3d.png'}")
    plt.close()


def compare_modes(results_simple, results_pe, theta_vals, save_dir):
    """Compare simple and pursuit-evasion modes."""
    fig, axes = plt.subplots(len(theta_vals), 3, figsize=(15, 5*len(theta_vals)))

    if len(theta_vals) == 1:
        axes = axes.reshape(1, -1)

    for i, theta in enumerate(theta_vals):
        # Find results for this theta
        r_simple = next(r for r in results_simple if r['theta'] == theta)
        r_pe = next(r for r in results_pe if r['theta'] == theta)

        # Simple mode
        ax = axes[i, 0]
        cs1 = ax.contourf(r_simple['X'], r_simple['Z'], r_simple['V'],
                         levels=20, cmap='RdBu_r', vmin=-1, vmax=1)
        ax.contour(r_simple['X'], r_simple['Z'], r_simple['V'],
                  levels=[0], colors='black', linewidths=2)
        ax.set_title(f'Simple (θ={np.degrees(theta):.1f}°)')
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.set_aspect('equal')
        plt.colorbar(cs1, ax=ax)

        # Pursuit-evasion mode
        ax = axes[i, 1]
        cs2 = ax.contourf(r_pe['X'], r_pe['Z'], r_pe['V'],
                         levels=20, cmap='RdBu_r', vmin=-1, vmax=1)
        ax.contour(r_pe['X'], r_pe['Z'], r_pe['V'],
                  levels=[0], colors='black', linewidths=2)
        ax.set_title(f'Pursuit-Evasion (θ={np.degrees(theta):.1f}°)')
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.set_aspect('equal')
        plt.colorbar(cs2, ax=ax)

        # Overlay comparison
        ax = axes[i, 2]
        ax.contour(r_simple['X'], r_simple['Z'], r_simple['V'],
                  levels=[0], colors='blue', linewidths=2, label='Simple')
        ax.contour(r_pe['X'], r_pe['Z'], r_pe['V'],
                  levels=[0], colors='red', linewidths=2, label='Pursuit-Evasion')
        circle = plt.Circle((0, 0), 0.5, fill=False, color='green',
                          linestyle='--', linewidth=2, label='Target')
        ax.add_patch(circle)
        ax.set_title(f'Comparison (θ={np.degrees(theta):.1f}°)')
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.tight_layout()
    plt.savefig(save_dir / "dubins_mode_comparison.png", dpi=150, bbox_inches='tight')
    print(f"Saved: {save_dir / 'dubins_mode_comparison.png'}")
    plt.close()


def main():
    """Main demo script."""
    print("="*60)
    print("DUBINS CAR BRT COMPUTATION - SAMPLING METHOD")
    print("="*60)

    # Parameters
    theta_vals = [0.0, np.pi/6, np.pi/4, np.pi/3, np.pi/2]
    theta_vals_comp = [0.0, np.pi/4, np.pi/2]  # For comparison plot
    grid_res = 40
    domain = (-2.0, 2.0)

    save_dir = Path(__file__).parent / "results"
    save_dir.mkdir(exist_ok=True)

    # Run simple mode
    results_simple = run_mode('simple', theta_vals, grid_res, domain)
    plot_results(results_simple, 'simple', save_dir)

    # Run pursuit-evasion mode
    results_pe = run_mode('pursuit_evasion', theta_vals, grid_res, domain)
    plot_results(results_pe, 'pursuit_evasion', save_dir)

    # Compare modes
    print("\n" + "="*60)
    print("COMPARING MODES")
    print("="*60)
    compare_modes(results_simple, results_pe, theta_vals_comp, save_dir)

    print("\n" + "="*60)
    print("DEMO COMPLETE")
    print("="*60)
    print(f"\nResults saved to: {save_dir}")


if __name__ == "__main__":
    main()
