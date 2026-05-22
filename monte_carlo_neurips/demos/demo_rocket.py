#!/usr/bin/env python
"""Demo script for rocket BRT computation using sampling method.

Visualizes:
1. Value function at different theta slices
2. Zero-level sets (BRT boundary)
3. Convergence history
"""

import os, sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D

# sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.sampling_engine import SolverConfig
from dynamics.rocket_python import RocketsSolver


def main():
    """Main demo script."""
    print("="*60)
    print("ROCKET BRT COMPUTATION - SAMPLING METHOD")
    print("="*60)

    # Configuration
    config = SolverConfig(
        delta=0.1,
        num_samples=10_000,
        max_iters=20,
        tol=1e-3,
        t_start=0.0,
        t_end=1.0,
        seed=123,
    )

    solver = RocketsSolver(config=config)

    # Compute BRT at different theta slices
    theta_vals = [0.0, np.pi/6, np.pi/4, np.pi/3, np.pi/2]
    grid_res = 40
    domain = (-3.0, 3.0)

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

    # Create visualization
    save_dir = Path(__file__).parent / "results"
    save_dir.mkdir(exist_ok=True)

    # Plot 1: Value functions at different theta
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    fontdict = {'fontsize':18, 'fontweight':'bold'}
    for i, result in enumerate(results[:6]):
        ax = axes[i]
        X = result['X']
        Z = result['Z']
        V = result['V']
        theta = result['theta']

        # Contour plot
        cs = ax.contourf(X, Z, V, levels=20, cmap='RdBu_r', vmin=-2, vmax=2)
        ax.contour(X, Z, V, levels=[0], colors='black', linewidths=2)

        # Add target circle
        circle = plt.Circle((0, 0), 1.5, fill=False, color='green',
                          linestyle='--', linewidth=2, label='Target')
        ax.add_patch(circle)

        ax.set_title(f'θ = {np.degrees(theta):.1f}°')
        ax.set_xlabel('x (m)', fontdict=fontdict)
        ax.set_ylabel('y (m)', fontdict=fontdict)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        plt.colorbar(cs, ax=ax, label='Value')

    plt.tight_layout()
    plt.savefig(save_dir / "rocket_brt_slices.png", dpi=150, bbox_inches='tight')
    print(f"\nSaved: {save_dir / 'rocket_brt_slices.png'}")
    plt.close()

    # Plot 2: Convergence history
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    for result in results:
        theta = result['theta']
        history = result['history']
        ax.semilogy(history, label=f'θ={np.degrees(theta):.1f}°', marker='o')

    ax.set_xlabel('Iteration')
    ax.set_ylabel('Relative Residual')
    ax.set_title('Convergence History')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Plot 3: Zero-level sets overlay
    ax = axes[1]
    for result in results:
        theta = result['theta']
        X = result['X']
        Z = result['Z']
        V = result['V']

        ax.contour(X, Z, V, levels=[0], linewidths=2,
                  label=f'θ={np.degrees(theta):.1f}°')

    # Add target
    circle = plt.Circle((0, 0), 1.5, fill=False, color='green',
                       linestyle='--', linewidth=2, label='Target')
    ax.add_patch(circle)

    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_title('BRT Boundaries (Zero-Level Sets)')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=8)

    plt.tight_layout()
    plt.savefig(save_dir / "rocket_convergence.png", dpi=150, bbox_inches='tight')
    print(f"Saved: {save_dir / 'rocket_convergence.png'}")
    plt.close()

    # Plot 3: 3D visualization of a single slice
    result = results[2]  # Use theta = pi/4
    X = result['X']
    Z = result['Z']
    V = result['V']
    theta = result['theta']

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(X, Z, V, cmap='RdBu_r', alpha=0.8,
                          vmin=-2, vmax=2, edgecolor='none')

    # Add zero-level contour
    ax.contour(X, Z, V, levels=[0], colors='black', linewidths=3,
              offset=V.min())

    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_zlabel('Value')
    ax.set_title(f'Value Function (θ = {np.degrees(theta):.1f}°)')
    fig.colorbar(surf, ax=ax, shrink=0.5, label='Value')

    plt.savefig(save_dir / "rocket_3d_value.png", dpi=150, bbox_inches='tight')
    print(f"Saved: {save_dir / 'rocket_3d_value.png'}")
    plt.close()

    print("\n" + "="*60)
    print("DEMO COMPLETE")
    print("="*60)
    print(f"\nResults saved to: {save_dir}")


if __name__ == "__main__":
    main()
