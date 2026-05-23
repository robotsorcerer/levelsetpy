#!/usr/bin/env python
"""3D Dubins pursuit-evasion: MC Cole-Hopf vs levelsetpy grid solver.

Compares the sampling-based solver to the levelsetpy finite-difference
solver on the Merz (1972) two-Dubins pursuit-evasion problem in
relative coordinates.

State: (x1, x2, x3) -- relative position and heading.
Target: cylinder of radius 0.5 (axis-aligned along x3/theta).

The sampling solver solves the *viscous* HJ PDE (with delta > 0),
while levelsetpy solves the *inviscid* PDE (delta = 0).  By
Crandall-Lions, the discrepancy is O(sqrt(delta)).

Produces: examples/dubins_3d_comparison.png
"""

import sys
import os
import copy
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, "/home/lex/Documents/ML-Control-Rob/control/levelsetpy")

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from math import pi
from functools import partial
from scipy.interpolate import RegularGridInterpolator

from src.config import SolverConfig
from src.hamiltonians import DubinsRelativeHamiltonian
from src.initial_conditions import cylinder_cost
from src.hj_sampler import HJReachabilitySampler


# ═══════════════════════════════════════════════════════════════════════
#  Global plot style: bold, large fonts
# ═══════════════════════════════════════════════════════════════════════
FONTDICT = {"fontsize": 24, "fontweight": "bold"}
TITLE_FONTDICT = {"fontsize": 22, "fontweight": "bold"}
plt.rcParams.update({
    "font.weight": "bold",
    "axes.labelweight": "bold",
    "axes.titleweight": "bold",
    "axes.labelsize": 24,
    "axes.titlesize": 20,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "figure.titlesize": 22,
    "figure.titleweight": "bold",
})

# ═══════════════════════════════════════════════════════════════════════
#  Parameters (shared between both solvers)
# ═══════════════════════════════════════════════════════════════════════
V_P = 1.0          # pursuer linear speed
V_E = 1.0          # evader linear speed
W = 1.0            # angular speed bound (equal for both)
RADIUS = 0.5       # cylinder target radius
T_FINAL = 1.0      # backward time horizon

# Domain
GRID_MIN = np.array([-4.0, -4.0, -pi])
GRID_MAX = np.array([4.0, 4.0, pi])

# Grid resolution for levelsetpy
GRID_N_LS = 45     # per axis (45^3 ≈ 91k points, tractable)

# Evaluation grid for MC solver (2D slices)
GRID_RES_MC = 40   # per spatial axis
SPATIAL_DOMAIN = (-4.0, 4.0)
THETA_SLICES = [-pi / 2, 0.0, pi / 2]

# MC solver config
DELTA = 0.08       # viscosity parameter
cfg = SolverConfig(
    delta=DELTA,
    num_samples=20_000,
    max_quasi_iters=15,
    quasi_tol=1e-5,
    t_start=0.0,
    t_end=T_FINAL,
    seed=42,
)


# ═══════════════════════════════════════════════════════════════════════
#  1. levelsetpy grid solver
# ═══════════════════════════════════════════════════════════════════════

def run_levelsetpy():
    """Run levelsetpy on the 3D Dubins relative pursuit-evasion problem."""
    import torch
    from levelsetpy.grids import createGrid
    from levelsetpy.initialconditions import shapeCylinder
    from levelsetpy.dynamicalsystems import DubinsVehicleRel
    from levelsetpy.utilities import Bundle
    from levelsetpy.explicitintegration.term import (
        termLaxFriedrichs, termRestrictUpdate,
    )
    from levelsetpy.explicitintegration.integration import odeCFL3, odeCFLset
    from levelsetpy.explicitintegration.dissipation import artificialDissipationGLF
    from levelsetpy.spatialderivative import upwindFirstENO2

    print("=== levelsetpy grid solver ===")

    # Grid with periodic theta dimension
    gmin = GRID_MIN.reshape(3, 1).astype(np.float64)
    gmax = GRID_MAX.reshape(3, 1).astype(np.float64)
    g = createGrid(gmin, gmax, GRID_N_LS, pdDims=2)

    # Initial condition: cylinder (axis along theta)
    phi0 = shapeCylinder(g, axis_align=2, center=np.zeros((3, 1)), radius=RADIUS)

    # Convert grid to torch (DubinsVehicleRel uses self.grid.xs with torch ops)
    g_torch = copy.deepcopy(g)
    g_torch.xs = [torch.as_tensor(x) for x in g.xs]

    # Dynamics (must receive torch grid since hamiltonian uses self.grid.xs)
    dubins = DubinsVehicleRel(g_torch, u_bound=V_P, w_bound=W)

    finite_diff_data = Bundle(dict(
        innerFunc=termLaxFriedrichs,
        innerData=Bundle({
            "grid": g_torch,
            "hamFunc": dubins.hamiltonian,
            "partialFunc": dubins.dissipation,
            "dissFunc": artificialDissipationGLF,
            "CoStateCalc": upwindFirstENO2,
        }),
        positive=False,
    ))

    small = 100 * np.finfo(np.float64).eps
    options = Bundle(dict(
        factorCFL=0.75, stats="on",
        maxStep=1e10, singleStep="off",
    ))

    y0 = torch.as_tensor(phi0.flatten())
    cur_time = 0.0
    step_time = T_FINAL / 5.0

    t0 = time.time()
    while T_FINAL - cur_time > small * T_FINAL:
        t_span = np.hstack([cur_time, min(T_FINAL, cur_time + step_time)])
        t, y, finite_diff_data = odeCFL3(
            termRestrictUpdate, t_span, y0,
            odeCFLset(options), finite_diff_data,
        )
        cur_time = t if np.isscalar(t) else t[-1]
        y0 = y
        print(f"  levelsetpy t = {cur_time:.3f}")
    elapsed_ls = time.time() - t0

    value_func = y.reshape(g.shape)
    if hasattr(value_func, "cpu"):
        value_func = value_func.cpu().numpy()

    print(f"  levelsetpy done in {elapsed_ls:.1f}s  "
          f"(grid {GRID_N_LS}^3 = {GRID_N_LS**3} points)")

    return g, np.array(value_func), elapsed_ls


def interpolate_3d_slice(g_ls, value_func, x_pts, y_pts, theta_val):
    """Interpolate the 3D value function on a 2D (x, y) slice at fixed theta.

    Returns (len(x_pts), len(y_pts)) array.
    """
    # Build 1D coordinate vectors from grid
    x_coords = g_ls.vs[0].ravel()
    y_coords = g_ls.vs[1].ravel()
    t_coords = g_ls.vs[2].ravel()

    interp = RegularGridInterpolator(
        (x_coords, y_coords, t_coords), value_func,
        method="linear", bounds_error=False, fill_value=np.nan,
    )

    # Create evaluation points
    XX, YY = np.meshgrid(x_pts, y_pts, indexing="ij")
    TT = np.full_like(XX, theta_val)
    pts = np.stack([XX.ravel(), YY.ravel(), TT.ravel()], axis=-1)
    return interp(pts).reshape(XX.shape)


# ═══════════════════════════════════════════════════════════════════════
#  2. Sampling solver
# ═══════════════════════════════════════════════════════════════════════

def run_mc_solver(theta_val):
    """Run the MC Cole-Hopf solver on a 2D slice at fixed theta."""
    H = DubinsRelativeHamiltonian(v_p=V_P, v_e=V_E, w=W)
    g_fn = partial(cylinder_cost, axis_align=2, radius=RADIUS)

    xs = jnp.linspace(*SPATIAL_DOMAIN, GRID_RES_MC)
    X, Y = jnp.meshgrid(xs, xs, indexing="ij")

    theta_col = jnp.full((GRID_RES_MC * GRID_RES_MC, 1), theta_val)
    eval_points = jnp.concatenate(
        [X.ravel()[:, None], Y.ravel()[:, None], theta_col], axis=-1
    )

    sampler = HJReachabilitySampler(H, g_fn, cfg)
    t0 = time.time()
    v, history = sampler.solve_quasi_linear(eval_points, 0.0)
    elapsed = time.time() - t0

    return X, Y, v.reshape(GRID_RES_MC, GRID_RES_MC), history, elapsed


# ═══════════════════════════════════════════════════════════════════════
#  3. Run both and compare
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    out_dir = "/home/lex/Documents/Papers/MoluxLabs/Neurips2026/HJ_Gauss/figures"
    os.makedirs(out_dir, exist_ok=True)
    # ── Run levelsetpy ──────────────────────────────────────────────
    g_ls, v_ls, time_ls = run_levelsetpy()

    # ── Run MC solver and compare ───────────────────────────────────
    n_slices = len(THETA_SLICES)
    fig, axes = plt.subplots(3, n_slices, figsize=(5 * n_slices, 14))

    # Shared spatial coordinates for interpolating the levelsetpy result
    xs_eval = np.linspace(*SPATIAL_DOMAIN, GRID_RES_MC)

    total_mc_time = 0.0
    for col, theta_val in enumerate(THETA_SLICES):
        print(f"\n--- theta = {theta_val:.2f} ---")

        # MC solver
        print(f"  MC solver ...")
        X, Y, V_mc, history, elapsed_mc = run_mc_solver(theta_val)
        total_mc_time += elapsed_mc
        print(f"  MC done in {elapsed_mc:.1f}s ({len(history)} iters)")

        # Interpolate levelsetpy onto the same grid
        V_ref = interpolate_3d_slice(g_ls, v_ls, xs_eval, xs_eval, theta_val)

        # Compute error (excluding NaN boundary cells)
        mask = np.isfinite(V_ref)
        diff = np.abs(np.array(V_mc) - V_ref)
        l_inf = np.nanmax(diff[mask]) if mask.any() else float("nan")
        l2_rel = (np.sqrt(np.nanmean(diff[mask] ** 2))
                  / max(np.sqrt(np.nanmean(V_ref[mask] ** 2)), 1e-12))
        print(f"  L_inf error = {l_inf:.3f},  L2_rel = {l2_rel:.3f}")
        print(f"  Expected O(sqrt(delta)) = {np.sqrt(DELTA):.4f}")

        X_np, Y_np = np.array(X), np.array(Y)

        # ── Row 0: levelsetpy ───────────────────────────────────
        ax = axes[0, col]
        cf = ax.contourf(X_np, Y_np, V_ref, levels=20, cmap="RdBu_r")
        ax.contour(X_np, Y_np, V_ref, levels=[0.0], colors="k", linewidths=2.5)
        ax.set_title(rf"$\theta={theta_val:.2f}$",
                     fontdict=TITLE_FONTDICT)
        # ax.set_xlabel(r"$\mathbf{x_1}$ (m)", fontdict=FONTDICT)
        ax.set_ylabel(r"$\mathbf{x_2}$ (m)", fontdict=FONTDICT)
        ax.set_aspect("equal")

        # ── Row 1: MC solver ───────────────────────────────────
        ax = axes[1, col]
        ax.contourf(X_np, Y_np, np.array(V_mc), levels=20, cmap="RdBu_r")
        ax.contour(X_np, Y_np, np.array(V_mc), levels=[0.0], colors="k",
                   linewidths=2.5)
        ax.set_title(
            rf"$\delta$={DELTA}$"
            f"\n{len(history)} iters, {elapsed_mc:.1f}s",
            fontdict=TITLE_FONTDICT,
        )
        # ax.set_xlabel(r"$\mathbf{x_1}$ (m)", fontdict=FONTDICT)
        ax.set_ylabel(r"$\mathbf{x_2}$ (m)", fontdict=FONTDICT)
        ax.set_aspect("equal")

        # ── Row 2: |error| ─────────────────────────────────────
        ax = axes[2, col]
        err_plot = ax.contourf(X_np, Y_np, diff, levels=20, cmap="hot_r")
        cb = plt.colorbar(err_plot, ax=ax, fraction=0.046)
        cb.ax.tick_params(labelsize=10)
        ax.contour(X_np, Y_np, V_ref, levels=[0.0], colors="cyan",
                   linewidths=2, linestyles="--")
        ax.set_title(
            rf"L$_\infty$={l_inf:.3f}  L$_2$={l2_rel:.3f}",
            fontdict=TITLE_FONTDICT,
        )
        ax.set_xlabel(r"$\mathbf{x_1}$ (m)", fontdict=FONTDICT)
        ax.set_ylabel(r"$\mathbf{x_2}$ (m)", fontdict=FONTDICT)
        ax.set_aspect("equal")

    axes[0, 0].set_ylabel(r"$\mathbf{x_2}$ (m)" + "\n(levelsetpy)",
                          fontdict=FONTDICT)
    axes[1, 0].set_ylabel(r"$\mathbf{x_2}$ (m)" + "\n(MC sampling)",
                          fontdict=FONTDICT)
    axes[2, 0].set_ylabel(r"$\mathbf{x_2}$ (m)" + "\n(|error|)",
                          fontdict=FONTDICT)

    fig.suptitle(
        f"Dubins Pursuit-Evasion BRS: MC Cole-Hopf vs levelsetpy\n"
        f"Grid: {GRID_N_LS}³ | MC: {cfg.num_samples} samples, "
        f"$\\delta={DELTA}$ | T={T_FINAL}",
        fontsize=22, fontweight="bold",
    )
    fig.tight_layout()
    out = os.path.join(out_dir, "dubins_3d_comparison.jpg")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nSaved → {out}")
    print(f"\nTiming: levelsetpy={time_ls:.1f}s, MC total={total_mc_time:.1f}s")
    plt.close(fig)
