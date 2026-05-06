#!/usr/bin/env python
"""3D Two-Rockets pursuit-evasion: MC Cole-Hopf vs levelsetpy grid solver.

Compares the sampling-based solver to the levelsetpy finite-difference
solver on the two-rockets pursuit-evasion problem in relative coordinates.

State: (x, z, theta) -- relative position and thrust inclination.
Target: cylinder of radius 1.5 (axis-aligned along theta).

Outputs:
  - examples/rockets_3d_slices.png   (2D slice comparison)
  - examples/rockets_3d_brt.png      (3D BRT isosurface comparison)
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
from skimage import measure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from src.config import SolverConfig
from src.hamiltonians.base import Hamiltonian
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
#  Hamiltonian matching levelsetpy's RocketSystemRel exactly
# ═══════════════════════════════════════════════════════════════════════

class RocketsRelativeLSHamiltonian(Hamiltonian):
    """Rockets Hamiltonian matching levelsetpy RocketSystemRel.

    levelsetpy formula (with u_p = -u_bound^2):
        H = -a*cos(x3)*p1 + (g-a-a*sin(x3))*p2
            + u_bound^2 * |p1*x1+p3| - u_bound^2 * |p2*x1+p3|
    """

    def __init__(self, a=1.0, g=2.0, u_bound=1.0, smoothing_eps=1e-4):
        self.a = a
        self.grav = g
        self.u_eff = u_bound ** 2  # levelsetpy squares u_bound
        self.eps = smoothing_eps

    def __call__(self, t, x, p):
        p1, p2, p3 = p[..., 0], p[..., 1], p[..., 2]
        x1 = x[..., 0]
        x3 = x[..., 2]
        smooth_abs = lambda z: jnp.sqrt(z ** 2 + self.eps ** 2)

        term1 = -self.a * jnp.cos(x3) * p1
        term2 = (self.grav - self.a - self.a * jnp.sin(x3)) * p2
        term3 = self.u_eff * smooth_abs(p1 * x1 + p3)
        term4 = -self.u_eff * smooth_abs(p2 * x1 + p3)
        return term1 + term2 + term3 + term4

    @property
    def state_dim(self) -> int:
        return 3


# ═══════════════════════════════════════════════════════════════════════
#  Shared parameters
# ═══════════════════════════════════════════════════════════════════════
A_THRUST = 1.0
GRAV = 2.0
U_BOUND = 1.0
RADIUS = 1.5
T_FINAL = 0.5

GRID_MIN = np.array([-5.0, -5.0, -pi])
GRID_MAX = np.array([5.0, 5.0, pi])

GRID_N_LS = 45        # levelsetpy grid per axis
GRID_N_MC_2D = 40     # MC 2D slices per spatial axis
GRID_N_MC_3D = 25     # MC 3D grid per axis (for marching cubes)
SPATIAL_DOMAIN = (-5.0, 5.0)
THETA_SLICES = [-pi / 2, 0.0, pi / 2]

DELTA = 0.08
MC_CFG = SolverConfig(
    delta=DELTA,
    num_samples=14_000,
    max_quasi_iters=12,
    quasi_tol=1e-5,
    t_start=0.0,
    t_end=T_FINAL,
    seed=42,
)


# ═══════════════════════════════════════════════════════════════════════
#  1. levelsetpy grid solver
# ═══════════════════════════════════════════════════════════════════════

def run_levelsetpy():
    """Run levelsetpy on the 3D rockets problem."""
    import torch
    from levelsetpy.grids import createGrid
    from levelsetpy.initialconditions import shapeCylinder
    from levelsetpy.dynamicalsystems import RocketSystemRel
    from levelsetpy.utilities import Bundle
    from levelsetpy.explicitintegration.term import (
        termLaxFriedrichs, termRestrictUpdate,
    )
    from levelsetpy.explicitintegration.integration import odeCFL3, odeCFLset
    from levelsetpy.explicitintegration.dissipation import artificialDissipationGLF
    from levelsetpy.spatialderivative import upwindFirstENO2

    print("=== levelsetpy grid solver ===")

    gmin = GRID_MIN.reshape(3, 1).astype(np.float64)
    gmax = GRID_MAX.reshape(3, 1).astype(np.float64)
    g = createGrid(gmin, gmax, GRID_N_LS, pdDims=2)

    phi0 = shapeCylinder(g, axis_align=2, center=np.zeros((3, 1)), radius=RADIUS)

    # Convert grid to torch (required by RocketSystemRel internals)
    g_torch = copy.deepcopy(g)
    g_torch.xs = [torch.as_tensor(x) for x in g.xs]

    rockets = RocketSystemRel(g_torch, u_bound=U_BOUND, a=A_THRUST, g=GRAV)

    finite_diff_data = Bundle(dict(
        innerFunc=termLaxFriedrichs,
        innerData=Bundle({
            "grid": g_torch,
            "hamFunc": rockets.hamiltonian,
            "partialFunc": rockets.dissipation,
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
    elapsed = time.time() - t0

    value_func = y.reshape(g.shape)
    if hasattr(value_func, "cpu"):
        value_func = value_func.cpu().numpy()

    print(f"  done in {elapsed:.1f}s (grid {GRID_N_LS}^3 = {GRID_N_LS**3})")
    return g, np.array(value_func), elapsed


def interp_3d_slice(g_ls, val, x_pts, y_pts, theta_val):
    """Interpolate levelsetpy 3D value onto a 2D (x,z) slice at fixed theta."""
    x_c = g_ls.vs[0].ravel()
    y_c = g_ls.vs[1].ravel()
    t_c = g_ls.vs[2].ravel()
    interp = RegularGridInterpolator(
        (x_c, y_c, t_c), val,
        method="linear", bounds_error=False, fill_value=np.nan,
    )
    XX, YY = np.meshgrid(x_pts, y_pts, indexing="ij")
    TT = np.full_like(XX, theta_val)
    pts = np.stack([XX.ravel(), YY.ravel(), TT.ravel()], axis=-1)
    return interp(pts).reshape(XX.shape)


def interp_3d_volume(g_ls, val, x_pts, y_pts, t_pts):
    """Interpolate levelsetpy 3D value onto a regular 3D grid."""
    x_c = g_ls.vs[0].ravel()
    y_c = g_ls.vs[1].ravel()
    t_c = g_ls.vs[2].ravel()
    interp = RegularGridInterpolator(
        (x_c, y_c, t_c), val,
        method="linear", bounds_error=False, fill_value=np.nan,
    )
    XX, YY, TT = np.meshgrid(x_pts, y_pts, t_pts, indexing="ij")
    pts = np.stack([XX.ravel(), YY.ravel(), TT.ravel()], axis=-1)
    return interp(pts).reshape(XX.shape)


# ═══════════════════════════════════════════════════════════════════════
#  2. MC sampling solver
# ═══════════════════════════════════════════════════════════════════════

def run_mc_2d_slice(theta_val):
    """Evaluate MC solver on a 2D (x, z) slice at fixed theta."""
    H = RocketsRelativeLSHamiltonian(a=A_THRUST, g=GRAV, u_bound=U_BOUND)
    g_fn = partial(cylinder_cost, axis_align=2, radius=RADIUS)

    xs = jnp.linspace(*SPATIAL_DOMAIN, GRID_N_MC_2D)
    X, Z = jnp.meshgrid(xs, xs, indexing="ij")
    theta_col = jnp.full((GRID_N_MC_2D ** 2, 1), theta_val)
    eval_points = jnp.concatenate(
        [X.ravel()[:, None], Z.ravel()[:, None], theta_col], axis=-1,
    )

    sampler = HJReachabilitySampler(H, g_fn, MC_CFG)
    t0 = time.time()
    v, history = sampler.solve_quasi_linear(eval_points, 0.0)
    elapsed = time.time() - t0
    return X, Z, v.reshape(GRID_N_MC_2D, GRID_N_MC_2D), history, elapsed


def run_mc_3d_volume():
    """Evaluate MC solver on the full 3D grid for marching cubes."""
    H = RocketsRelativeLSHamiltonian(a=A_THRUST, g=GRAV, u_bound=U_BOUND)
    g_fn = partial(cylinder_cost, axis_align=2, radius=RADIUS)

    xs = jnp.linspace(*SPATIAL_DOMAIN, GRID_N_MC_3D)
    ts = jnp.linspace(-pi, pi, GRID_N_MC_3D)
    XX, ZZ, TT = jnp.meshgrid(xs, xs, ts, indexing="ij")
    eval_points = jnp.stack(
        [XX.ravel(), ZZ.ravel(), TT.ravel()], axis=-1,
    )

    print(f"  MC 3D: {eval_points.shape[0]} points ...")
    sampler = HJReachabilitySampler(H, g_fn, MC_CFG)
    t0 = time.time()
    v, history = sampler.solve_quasi_linear(eval_points, 0.0)
    elapsed = time.time() - t0
    V = np.array(v.reshape(GRID_N_MC_3D, GRID_N_MC_3D, GRID_N_MC_3D))
    print(f"  MC 3D done in {elapsed:.1f}s ({len(history)} iters)")
    return V, elapsed


# ═══════════════════════════════════════════════════════════════════════
#  3. 3D BRT visualization (marching cubes)
# ═══════════════════════════════════════════════════════════════════════

def extract_mesh(volume, level, spacing, origin):
    """Extract zero-level isosurface and map vertices to physical coords."""
    try:
        verts, faces, normals, _ = measure.marching_cubes(
            volume, level=level, spacing=spacing,
        )
    except ValueError:
        return None, None
    # Shift vertices from voxel space to physical coordinates
    verts[:, 0] += origin[0]
    verts[:, 1] += origin[1]
    verts[:, 2] += origin[2]
    return verts, faces


def add_mesh_to_ax(ax, verts, faces, facecolor, edgecolor="k",
                   alpha=0.35, linewidth=0.1):
    """Add a Poly3DCollection to a 3D axes."""
    mesh = Poly3DCollection(
        verts[faces], alpha=alpha, linewidth=linewidth,
    )
    mesh.set_facecolor(facecolor)
    mesh.set_edgecolor(edgecolor)
    ax.add_collection3d(mesh)


def setup_3d_ax(ax, title, xlim, ylim, zlim):
    """Style a 3D axes for BRT plots."""
    ax.set_xlim3d(*xlim)
    ax.set_ylim3d(*ylim)
    ax.set_zlim3d(*zlim)
    ax.set_xlabel(r"$\mathbf{x}$ (m)", fontdict=FONTDICT, labelpad=8)
    ax.set_ylabel(r"$\mathbf{z}$ (m)", fontdict=FONTDICT, labelpad=8)
    ax.set_zlabel(r"$\theta$ (rad)", fontdict=FONTDICT, labelpad=8)
    ax.set_title(title, fontdict=TITLE_FONTDICT, pad=12)
    ax.tick_params(axis="both", which="major", labelsize=10)
    ax.view_init(elev=25, azim=-50)
    ax.grid(True)


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    out_dir = "/home/lex/Documents/Papers/MoluxLabs/Neurips2026/HJ_Gauss/figures"
    os.makedirs(out_dir, exist_ok=True)

    # ── Run levelsetpy ──────────────────────────────────────────────
    g_ls, v_ls, time_ls = run_levelsetpy()

    # ═════════════════════════════════════════════════════════════════
    #  PART A: 2D slice comparison
    # ═════════════════════════════════════════════════════════════════
    n_slices = len(THETA_SLICES)
    fig_slices, axes = plt.subplots(
        3, n_slices, figsize=(6 * n_slices, 16),
    )
    xs_eval = np.linspace(*SPATIAL_DOMAIN, GRID_N_MC_2D)

    total_mc_2d = 0.0
    for col, theta_val in enumerate(THETA_SLICES):
        print(f"\n--- theta = {theta_val:.2f} ---")

        # MC solver
        print("  MC 2D slice ...")
        X, Z, V_mc, history, elapsed = run_mc_2d_slice(theta_val)
        total_mc_2d += elapsed
        print(f"  done {elapsed:.1f}s ({len(history)} iters)")

        # Interpolate levelsetpy onto same grid
        V_ref = interp_3d_slice(g_ls, v_ls, xs_eval, xs_eval, theta_val)

        # Error metrics
        mask = np.isfinite(V_ref)
        diff = np.abs(np.array(V_mc) - V_ref)
        l_inf = np.nanmax(diff[mask]) if mask.any() else float("nan")
        l2_rel = (np.sqrt(np.nanmean(diff[mask] ** 2))
                  / max(np.sqrt(np.nanmean(V_ref[mask] ** 2)), 1e-12))
        print(f"  L_inf = {l_inf:.4f},  L2_rel = {l2_rel:.4f}")

        X_np, Z_np = np.array(X), np.array(Z)

        # Row 0: levelsetpy
        ax = axes[0, col]
        ax.contourf(X_np, Z_np, V_ref, levels=20, cmap="RdBu_r")
        ax.contour(X_np, Z_np, V_ref, levels=[0.0], colors="k", linewidths=2.5)
        ax.set_title(rf"levelsetpy  $\theta={theta_val:.2f}$",
                     fontdict=TITLE_FONTDICT)
        ax.set_xlabel(r"$\mathbf{x}$ (m)", fontdict=FONTDICT)
        ax.set_ylabel(r"$\mathbf{z}$ (m)", fontdict=FONTDICT)
        ax.set_aspect("equal")

        # Row 1: MC solver
        ax = axes[1, col]
        ax.contourf(X_np, Z_np, np.array(V_mc), levels=20, cmap="RdBu_r")
        ax.contour(X_np, Z_np, np.array(V_mc), levels=[0.0], colors="k",
                   linewidths=2.5)
        ax.set_title(
            rf"MC ($\delta$={DELTA})  $\theta={theta_val:.2f}$"
            f"\n{len(history)} iters, {elapsed:.1f}s",
            fontdict=TITLE_FONTDICT,
        )
        ax.set_xlabel(r"$\mathbf{x}$ (m)", fontdict=FONTDICT)
        ax.set_ylabel(r"$\mathbf{z}$ (m)", fontdict=FONTDICT)
        ax.set_aspect("equal")

        # Row 2: |error|
        ax = axes[2, col]
        err_plot = ax.contourf(X_np, Z_np, diff, levels=20, cmap="hot_r")
        cb = plt.colorbar(err_plot, ax=ax, fraction=0.046)
        cb.ax.tick_params(labelsize=10)
        ax.contour(X_np, Z_np, V_ref, levels=[0.0], colors="cyan",
                   linewidths=2, linestyles="--")
        ax.set_title(
            rf"|error|  L$_\infty$={l_inf:.3f}  L$_2$={l2_rel:.3f}",
            fontdict=TITLE_FONTDICT,
        )
        ax.set_xlabel(r"$\mathbf{x}$ (m)", fontdict=FONTDICT)
        ax.set_ylabel(r"$\mathbf{z}$ (m)", fontdict=FONTDICT)
        ax.set_aspect("equal")

    axes[0, 0].set_ylabel(r"$\mathbf{z}$ (m)" + "\n(levelsetpy)",
                          fontdict=FONTDICT)
    axes[1, 0].set_ylabel(r"$\mathbf{z}$ (m)" + "\n(MC sampling)",
                          fontdict=FONTDICT)
    axes[2, 0].set_ylabel(r"$\mathbf{z}$ (m)" + "\n(|error|)",
                          fontdict=FONTDICT)

    fig_slices.suptitle(
        f"Two-Rockets BRT: MC Cole-Hopf vs levelsetpy\n"
        # f"a={A_THRUST}, g={GRAV}, Grid: {GRID_N_LS}³ | "
        f"MC: {MC_CFG.num_samples} samples, δ={DELTA} | T={T_FINAL}",
        fontsize=16, fontweight="bold",
    )
    fig_slices.tight_layout()
    out_slices = os.path.join(out_dir, "rockets_3d_slices.jpg")
    fig_slices.savefig(out_slices, dpi=150, bbox_inches="tight")
    print(f"\nSaved 2D slices → {out_slices}")
    plt.close(fig_slices)

    # ═════════════════════════════════════════════════════════════════
    #  PART B: 3D BRT isosurface comparison
    # ═════════════════════════════════════════════════════════════════
    print("\n=== 3D BRT isosurface comparison ===")

    # Compute MC 3D volume
    V_mc_3d, time_mc_3d = run_mc_3d_volume()

    # Interpolate levelsetpy onto the same 3D grid
    xs_3d = np.linspace(*SPATIAL_DOMAIN, GRID_N_MC_3D)
    ts_3d = np.linspace(-pi, pi, GRID_N_MC_3D)
    V_ls_3d = interp_3d_volume(g_ls, v_ls, xs_3d, xs_3d, ts_3d)

    # Grid spacing for marching cubes
    dx = (SPATIAL_DOMAIN[1] - SPATIAL_DOMAIN[0]) / (GRID_N_MC_3D - 1)
    dt = 2 * pi / (GRID_N_MC_3D - 1)
    spacing = (dx, dx, dt)
    origin = (SPATIAL_DOMAIN[0], SPATIAL_DOMAIN[0], -pi)

    # Extract meshes at zero level
    verts_ls, faces_ls = extract_mesh(V_ls_3d, level=0.0,
                                      spacing=spacing, origin=origin)
    verts_mc, faces_mc = extract_mesh(V_mc_3d, level=0.0,
                                      spacing=spacing, origin=origin)

    # ── Figure: side-by-side 3D BRT ────────────────────────────────
    fig_3d = plt.figure(figsize=(20, 9))

    # Axis limits (union of both meshes)
    all_verts = []
    if verts_ls is not None:
        all_verts.append(verts_ls)
    if verts_mc is not None:
        all_verts.append(verts_mc)

    if all_verts:
        combined = np.vstack(all_verts)
        xlim = (combined[:, 0].min() - 0.3, combined[:, 0].max() + 0.3)
        ylim = (combined[:, 1].min() - 0.3, combined[:, 1].max() + 0.3)
        zlim = (combined[:, 2].min() - 0.1, combined[:, 2].max() + 0.1)
    else:
        xlim = SPATIAL_DOMAIN
        ylim = SPATIAL_DOMAIN
        zlim = (-pi, pi)

    # Left: levelsetpy BRT
    ax1 = fig_3d.add_subplot(1, 2, 1, projection="3d")
    if verts_ls is not None and faces_ls is not None:
        add_mesh_to_ax(ax1, verts_ls, faces_ls,
                       facecolor="royalblue", edgecolor="midnightblue",
                       alpha=0.35, linewidth=0.15)
    setup_3d_ax(ax1, "levelsetpy (inviscid)", xlim, ylim, zlim)

    # Right: MC BRT
    ax2 = fig_3d.add_subplot(1, 2, 2, projection="3d")
    if verts_mc is not None and faces_mc is not None:
        add_mesh_to_ax(ax2, verts_mc, faces_mc,
                       facecolor="tomato", edgecolor="darkred",
                       alpha=0.35, linewidth=0.15)
    setup_3d_ax(ax2, rf"MC Cole-Hopf ($\delta$={DELTA})", xlim, ylim, zlim)

    fig_3d.suptitle(
        f"Two-Rockets 3D Backward Reachable Tube (zero level set)\n"
        #f"a={A_THRUST}, g={GRAV}, T={T_FINAL} | "
        f"levelsetpy: {GRID_N_LS}³ grid | "
        f"MC: {GRID_N_MC_3D}³ grid, {MC_CFG.num_samples} samples",
        fontsize=26, fontweight="bold",
    )
    fig_3d.tight_layout()
    out_3d = os.path.join(out_dir, "rockets_3d_brt.jpg")
    fig_3d.savefig(out_3d, dpi=150, bbox_inches="tight")
    print(f"Saved 3D BRT → {out_3d}")
    plt.close(fig_3d)

    # ── Figure: overlay (both meshes in same axes) ─────────────────
    fig_overlay = plt.figure(figsize=(12, 10))
    ax_ov = fig_overlay.add_subplot(1, 1, 1, projection="3d")
    if verts_ls is not None and faces_ls is not None:
        add_mesh_to_ax(ax_ov, verts_ls, faces_ls,
                       facecolor="royalblue", edgecolor="midnightblue",
                       alpha=0.25, linewidth=0.1)
    if verts_mc is not None and faces_mc is not None:
        add_mesh_to_ax(ax_ov, verts_mc, faces_mc,
                       facecolor="tomato", edgecolor="darkred",
                       alpha=0.25, linewidth=0.1)
    setup_3d_ax(ax_ov, "BRT Overlay: levelsetpy (blue) vs MC (red)",
                xlim, ylim, zlim)

    # Legend proxy artists
    from matplotlib.patches import Patch
    legend_elems = [
        Patch(facecolor="royalblue", alpha=0.5, label="levelsetpy (inviscid)"),
        Patch(facecolor="tomato", alpha=0.5, label=f"MC (δ={DELTA})"),
    ]
    ax_ov.legend(handles=legend_elems, loc="upper left",
                 fontsize=12, framealpha=0.9)

    fig_overlay.suptitle(
        f"Two-Rockets BRT Overlay",
        fontsize=16, fontweight="bold",
    )
    fig_overlay.tight_layout()
    out_overlay = os.path.join(out_dir, "rockets_3d_overlay.jpg")
    fig_overlay.savefig(out_overlay, dpi=150, bbox_inches="tight")
    print(f"Saved overlay → {out_overlay}")
    plt.close(fig_overlay)

    print(f"\nTiming summary:")
    print(f"  levelsetpy:  {time_ls:.1f}s")
    print(f"  MC 2D total: {total_mc_2d:.1f}s")
    print(f"  MC 3D:       {time_mc_3d:.1f}s")
