#!/usr/bin/env python
"""Publication-grade figures for HJ-Gauss murmuration safety certification.

Standalone (lives OUTSIDE the repo). Computes BRT value grids with the 4D
aerial-murmuration Hamiltonian (Molu et al. IJRR23) and a configurable
union-of-capture-cylinders terminal cost, then renders:

  1. brt_evolution.{jpg,pdf}        -- hero small-multiples of {v<=0}
  2. brt_tube_3d.{jpg,pdf}          -- stacked reachable tube
  3. phase_space_snapshot.{jpg,pdf} -- subsampled birds colored by heading
  4. topology_evolution.{jpg,pdf}   -- chi(t), beta_1(t), n_comp(t)

NaNs from expectation underflow in solve_quasi_linear are masked to a large
positive sentinel ("far outside the BRT") so contours/fills are clean.

Honest science: we report exactly what the solver produces.

Usage examples:
  python make_pub_figures.py --out-dir /media/lex/data/hjgauss/iter03 \
      --scenario ring --n-predators 7 --ring-radius 1.6 --r-capture 0.55 \
      --delta 0.08 --n-samples 600 --grid-res 96 --time-steps 16 \
      --n-birds 8000 --n-flocks 6 --label "ring cordon probe"

Actual usage signals for paper:

python make_pub_figures.py \
  --out-dir /media/lex/data/hjgauss/figures_v2 \
  --scenario ring --n-predators 7 --ring-radius 1.15 --r-capture 0.6 \
  --delta 0.18 --n-samples 1600 --max-iters 7 --grid-res 128 \
  --time-steps 18 --n-snapshots 6 --n-birds 7998 --n-flocks 6 \
  --extent 3.5 --smooth-sigma 1.5
"""

import argparse
import json
import os
import time
import warnings
from os.path import join

os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.90")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Circle
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ---------------------------------------------------------------------------
# Global publication font style (bold, large) -- matches the author's
# levelsetpy/visualization conventions (fontweight='bold'), scaled up for
# print legibility. Titles 26-30, axis labels ~22, ticks ~18, all bold.
# ---------------------------------------------------------------------------
FS_SUPTITLE = 30     # main (sup)title
FS_TITLE    = 22     # per-axis / subplot title
FS_LABEL    = 24     # x/y/z axis labels
FS_TICK     = 18     # tick labels
FS_LEGEND   = 18     # legend / colorbar label
matplotlib.rcParams.update({
    "font.weight":       "bold",
    "axes.titleweight":  "bold",
    "axes.labelweight":  "bold",
    "axes.titlesize":    FS_TITLE,
    "axes.labelsize":    FS_LABEL,
    "axes.linewidth":    1.8,
    "xtick.labelsize":   FS_TICK,
    "ytick.labelsize":   FS_TICK,
    "xtick.major.width": 1.8,
    "ytick.major.width": 1.8,
    "xtick.major.size":  7,
    "ytick.major.size":  7,
    "legend.fontsize":   FS_LEGEND,
    "mathtext.default":  "bf",   # bold math in titles/labels
})


def _bold_ticks(ax):
    """Force bold tick labels (rcParams font.weight does not always take)."""
    for lab in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
        lab.set_fontweight("bold")
        lab.set_fontsize(FS_TICK)
    if hasattr(ax, "get_zticklabels"):
        for lab in ax.get_zticklabels():
            lab.set_fontweight("bold")
            lab.set_fontsize(FS_TICK)


def _bold_cbar(cb, label):
    cb.set_label(label, fontsize=FS_LEGEND, fontweight="bold")
    for lab in cb.ax.get_yticklabels():
        lab.set_fontweight("bold")
        lab.set_fontsize(FS_TICK - 2)


def _bold_cbar_h(cb, label):
    """Bold styling for a horizontal colorbar (ticks on the x-axis)."""
    cb.set_label(label, fontsize=FS_LEGEND, fontweight="bold")
    for lab in cb.ax.get_xticklabels():
        lab.set_fontweight("bold")
        lab.set_fontsize(FS_TICK - 2)

# ---------------------------------------------------------------------------
# Repo wiring
# ---------------------------------------------------------------------------
REPO = "/home/lex/Documents/ML-Control-Rob/control/levelsetpy/monte_carlo"
import sys
sys.path.insert(0, REPO)

import jax
jax.config.update("jax_platform_name", "cpu")
import jax.numpy as jnp

from src.config import SolverConfig
from src.gpu_distribution import GPUDistributor
from src.hj_sampler import HJReachabilitySampler
from src.hamiltonians.murmuration import MurmuationHamiltonian4D
from src.topology import (
    brt_topology_signature,
    detect_phase_transitions,
    brt_radius_at_time,
)
from dynamics.murmuration_jax import FlockState, PredatorState

SENTINEL = 1.0e3  # value assigned to masked (NaN/inf) grid cells
NAN_WARN_FRAC = 0.05  # only warn when >5% of grid cells underflow to NaN/inf

# Map raw topology events to the swarm-phase vocabulary requested.
EVENT_TO_PHASE = {
    "cordon_formation": "cordon",
    "vacuole_nucleation": "vacuole",
    "flock_fragmentation": "fragmentation",
    "flash_expansion": "flash-expansion",
}


# ---------------------------------------------------------------------------
# Terminal cost: union of capture cylinders (one per predator)
# ---------------------------------------------------------------------------
def make_union_terminal_cost(centers_xy, r_capture):
    """Return g(x) = min_i ||x_{1:2} - c_i|| - r_capture (JAX-traceable).

    centers_xy : (P, 2) array of capture-cylinder centers in the (x1,x2) plane.
    A single center at the origin reproduces dynamics.murmuration_jax.terminal_cost_4d.
    """
    centers = jnp.asarray(centers_xy, dtype=jnp.float32)
    r = float(r_capture)

    def g(x):
        xy = x[:2]
        d = jnp.sqrt(jnp.sum((centers - xy[None, :]) ** 2, axis=1) + 1e-12)
        return jnp.min(d) - r

    return g


def smooth_masked(V, sigma):
    """Normalized-convolution Gaussian smoothing that ignores sentinel cells.

    Denoises Monte-Carlo speckle in the value grid without bleeding the
    +SENTINEL "far outside" cells into the finite interior. Returns a grid on
    the same sentinel convention.
    """
    from scipy.ndimage import gaussian_filter
    if sigma <= 0:
        return V
    finite = V < SENTINEL / 2
    filled = np.where(finite, V, 0.0)
    w = finite.astype(np.float64)
    num = gaussian_filter(filled, sigma, mode="nearest")
    den = gaussian_filter(w, sigma, mode="nearest")
    out = np.where(den > 1e-3, num / np.maximum(den, 1e-9), SENTINEL)
    return out


def build_value_grid(sampler, t, grid_res, x_extent, altitude=50.0, heading=0.0,
                     smooth_sigma=0.0):
    """Evaluate v on a 2D (x1,x2) grid at fixed altitude & heading.

    Returns (V, X1, X2, nan_frac). NaN/inf cells are replaced by +SENTINEL;
    MC speckle optionally denoised by masked Gaussian smoothing.
    """
    x1_min, x1_max, x2_min, x2_max = x_extent
    x1 = np.linspace(x1_min, x1_max, grid_res)
    x2 = np.linspace(x2_min, x2_max, grid_res)
    X1, X2 = np.meshgrid(x1, x2, indexing="xy")
    pts = np.stack(
        [X1.ravel(), X2.ravel(),
         np.full(grid_res * grid_res, altitude, np.float32),
         np.full(grid_res * grid_res, heading, np.float32)],
        axis=1,
    ).astype(np.float32)
    v_flat, _ = sampler.solve_quasi_linear(jnp.asarray(pts), float(t))
    V = np.array(v_flat, dtype=np.float64).reshape(grid_res, grid_res)
    bad = ~np.isfinite(V)
    V[bad] = SENTINEL
    nanf = bad.mean()
    if smooth_sigma > 0:
        V = smooth_masked(V, smooth_sigma)
    return V, X1, X2, nanf


# ---------------------------------------------------------------------------
# Scenario construction
# ---------------------------------------------------------------------------
def predator_centers(scenario, n_pred, ring_radius, seed=2026):
    if scenario == "origin":
        return np.zeros((1, 2), np.float32)
    if scenario == "ring":
        ang = np.linspace(0, 2 * np.pi, n_pred, endpoint=False)
        return np.stack([ring_radius * np.cos(ang), ring_radius * np.sin(ang)], 1).astype(np.float32)
    if scenario == "spread":
        rng = np.random.default_rng(seed)
        return rng.uniform(-ring_radius, ring_radius, size=(n_pred, 2)).astype(np.float32)
    raise ValueError(scenario)


def make_flocks(n_birds, n_flocks, x_extent, seed=2026):
    key = jax.random.PRNGKey(seed)
    n_per = max(n_birds // n_flocks, 1)
    x1_min, x1_max, x2_min, x2_max = x_extent
    flocks = []
    for fid in range(n_flocks):
        key, sk = jax.random.split(key)
        s = jax.random.uniform(
            sk, (n_per, 4),
            minval=jnp.array([x1_min, x2_min, 0.0, -jnp.pi]),
            maxval=jnp.array([x1_max, x2_max, 100.0, jnp.pi]),
        )
        flocks.append(FlockState(states=s, flock_id=fid))
    return flocks


# ---------------------------------------------------------------------------
# Phase labelling per snapshot (from topology + radius trajectory)
# ---------------------------------------------------------------------------
def label_phases(topos, radii):
    """Return a human-readable phase label per snapshot."""
    labels = []
    for i, tp in enumerate(topos):
        if tp.n_components == 0:
            labels.append("seed")
            continue
        if tp.betti_1 >= 1:
            labels.append("cordon")
            continue
        if tp.n_components > 1:
            labels.append("fragmentation")
            continue
        # single simply-connected component
        if i > 0 and topos[i - 1].betti_1 >= 1:
            labels.append("vacuole-collapse")
        elif i > 0 and radii[i] - radii[i - 1] >= 2.0:
            labels.append("flash-expansion")
        elif i > 0 and radii[i] > radii[i - 1] + 1e-9:
            labels.append("evasion")
        else:
            labels.append("cohesion")
    return labels


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def fig_brt_evolution(out_dir, snaps, x_extent, centers, r_capture, label):
    """Hero small-multiples: filled {v<=0}, bold v=0 contour, cylinders, phase."""
    n = len(snaps)
    ncol = min(n, 3)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(6.4 * ncol, 6.0 * nrow),
                             squeeze=False,
                             gridspec_kw=dict(wspace=0.28, hspace=0.30))
    x1_min, x1_max, x2_min, x2_max = x_extent
    # symmetric color scale around 0 from finite values
    finite = np.concatenate([s["V"][np.isfinite(s["V"]) & (s["V"] < SENTINEL / 2)].ravel()
                             for s in snaps])
    vmax = np.nanpercentile(np.abs(finite), 98) if finite.size else 1.0
    vmax = max(vmax, 0.5)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    for k, s in enumerate(snaps):
        ax = axes[k // ncol][k % ncol]
        V = s["V"]
        Vp = np.where(V >= SENTINEL / 2, vmax, V)  # clamp sentinel for display
        ax.pcolormesh(s["X1"], s["X2"], Vp, cmap="RdBu_r", norm=norm,
                      shading="auto", rasterized=True)
        # filled safe set {v<=0}
        ax.contourf(s["X1"], s["X2"], V, levels=[-1e9, 0.0],
                    colors=["#2b2b6f"], alpha=0.30)
        if np.nanmin(V) <= 0 <= np.nanmax(V):
            ax.contour(s["X1"], s["X2"], V, levels=[0.0],
                       colors="k", linewidths=3.2)
        # capture cylinders
        for c in centers:
            ax.add_patch(Circle((c[0], c[1]), r_capture, fill=False,
                                 ec="lime", lw=2.2, ls="--", alpha=0.9))
            ax.plot(c[0], c[1], marker="x", color="lime", ms=11, mew=3)
        ax.set_xlim(x1_min, x1_max)
        ax.set_ylim(x2_min, x2_max)
        ax.set_aspect("equal")
        ax.set_title(f"$\\tau$={s['btime']:.2f}  [{s['phase']}]\n"
                     f"$\\chi$={s['chi']:.0f}  $\\beta_1$={s['beta1']}  "
                     f"$n_c$={s['ncomp']}", fontsize=FS_TITLE, fontweight="bold")
        if k % ncol == 0:
            ax.set_ylabel("$x_2$ (m)", fontsize=FS_LABEL, fontweight="bold")
        if k // ncol == nrow - 1:
            ax.set_xlabel("$x_1$ (m)", fontsize=FS_LABEL, fontweight="bold")
        _bold_ticks(ax)
    for k in range(n, nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    sm = cm.ScalarMappable(norm=norm, cmap="RdBu_r")
    sm.set_array([])
    cb = fig.colorbar(sm, ax=axes.ravel().tolist(), shrink=0.85, pad=0.015)
    _bold_cbar(cb, "value $v(x)$")
    fig.suptitle("BRT zero-level-set evolution (backward time $\\tau$)",
                 fontsize=FS_SUPTITLE, fontweight="bold", y=1.02)
    _save(fig, out_dir, "brt_evolution")


def fig_brt_tube_3d(out_dir, snaps, x_extent, label):
    """Stack {v=0} boundary contours along a vertical backward-time axis."""
    fig = plt.figure(figsize=(16, 9))
    # Place the 3D axes so it fills almost the whole figure; a slim dedicated
    # colorbar axes sits at the bottom. This removes the large default-3D
    # whitespace so the swept tube fills the (tall) canvas.
    ax = fig.add_axes([0.0, 0.06, 1.0, 0.92], projection="3d")
    import matplotlib.contour as _mc  # noqa
    cmap = cm.get_cmap("viridis")
    btimes = [s["btime"] for s in snaps]
    bmin, bmax = min(btimes), max(btimes)
    # tighten horizontal limits to the support of the tube so it fills the box
    rmax = 0.0
    for s in snaps:
        V = s["V"]
        if np.nanmin(V) <= 0 <= np.nanmax(V):
            xy = np.sqrt(s["X1"] ** 2 + s["X2"] ** 2)[V <= 0]
            if xy.size:
                rmax = max(rmax, float(np.nanmax(xy)))
    lim = min(max(rmax * 1.15, 1.0), max(abs(x_extent[0]), abs(x_extent[1])))
    drew = 0
    for s in snaps:
        V = s["V"]
        if not (np.nanmin(V) <= 0 <= np.nanmax(V)):
            continue
        cs = plt.contour(s["X1"], s["X2"], V, levels=[0.0])
        col = cmap((s["btime"] - bmin) / (bmax - bmin + 1e-9))
        z = s["btime"]
        for seg in cs.allsegs[0]:
            if len(seg) > 2:
                ax.plot(seg[:, 0], seg[:, 1], zs=z, color=col, lw=3.4, alpha=0.95)
                drew += 1
        plt.close(cs.axes.figure if hasattr(cs, "axes") else plt.gcf())
    ax.set_xlabel("$x_1$ (m)", fontsize=FS_LABEL, fontweight="bold", labelpad=26)
    ax.set_ylabel("$x_2$ (m)", fontsize=FS_LABEL, fontweight="bold", labelpad=26)
    ax.set_zlabel("backward time $\\tau$", fontsize=FS_LABEL, fontweight="bold",
                  labelpad=20)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_box_aspect((1.0, 1.0, 1.55))   # tall: emphasize the swept tau axis
    ax.view_init(elev=14, azim=-58)
    ax.tick_params(axis="both", which="major", pad=6)
    _bold_ticks(ax)
    ax.set_title("Backward reachable tube (swept $v=0$ level set)",
                 fontsize=FS_SUPTITLE, fontweight="bold", pad=10, y=0.99)
    sm = cm.ScalarMappable(norm=plt.Normalize(bmin, bmax), cmap=cmap)
    sm.set_array([])
    cax = fig.add_axes([0.18, 0.045, 0.64, 0.015])
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    _bold_cbar_h(cb, "backward time $\\tau$")
    _save(fig, out_dir, "brt_tube_3d")
    return drew


def fig_phase_space(out_dir, flock_states, rep_snap, x_extent, centers, r_capture,
                    label, max_pts=2000):
    states = np.asarray(flock_states)
    if states.shape[0] > max_pts:
        idx = np.random.default_rng(0).choice(states.shape[0], max_pts, replace=False)
        states = states[idx]
    fig, ax = plt.subplots(figsize=(10.5, 9.5))
    sc = ax.scatter(states[:, 0], states[:, 1], c=states[:, 3], cmap="hsv",
                    s=16, alpha=0.65, vmin=-np.pi, vmax=np.pi,
                    edgecolors="none", rasterized=True)
    cb = plt.colorbar(sc, ax=ax)
    _bold_cbar(cb, "heading $\\theta$ (rad)")
    V = rep_snap["V"]
    if np.nanmin(V) <= 0 <= np.nanmax(V):
        ax.contour(rep_snap["X1"], rep_snap["X2"], V, levels=[0.0],
                   colors="k", linewidths=3.5)
        ax.contourf(rep_snap["X1"], rep_snap["X2"], V, levels=[-1e9, 0.0],
                    colors=["k"], alpha=0.10)
    for c in centers:
        ax.add_patch(Circle((c[0], c[1]), r_capture, fill=False, ec="red",
                             lw=2.6, ls="--"))
        ax.plot(c[0], c[1], marker="*", color="red", ms=22, mec="k", mew=1.5)
    ax.set_xlim(x_extent[0], x_extent[1])
    ax.set_ylim(x_extent[2], x_extent[3])
    ax.set_aspect("equal")
    ax.set_xlabel("$x_1$ (m)", fontsize=FS_LABEL, fontweight="bold")
    ax.set_ylabel("$x_2$ (m)", fontsize=FS_LABEL, fontweight="bold")
    ax.set_title(f"Phase-space snapshot at $\\tau$={rep_snap['btime']:.2f}\n"
                 f"({states.shape[0]} birds subsampled)",
                 fontsize=FS_TITLE, fontweight="bold", pad=12)
    _bold_ticks(ax)
    _save(fig, out_dir, "phase_space_snapshot")


def fig_topology(out_dir, btimes, chi, beta1, ncomp, events, label):
    fig, axes = plt.subplots(1, 3, figsize=(20, 6.0))
    x = np.asarray(btimes)
    panels = [
        (chi, "Euler Characteristic $\\chi(\\tau)$", "b-o"),
        (beta1, "First Betti Number $\\beta_1(\\tau)$ (Holes)", "r-s"),
        (ncomp, "Connected Components $n_c(\\tau)$", "g-^"),
    ]
    for j, (ax, (y, ttl, sty)) in enumerate(zip(axes, panels)):
        ax.plot(x, y, sty, ms=10, lw=3.0, mew=2.0)
        ax.set_xlabel("Backward Time $\\tau$", fontsize=FS_LABEL, fontweight="bold")
        if j == 0:
            ax.set_ylabel("Topological Invariant", fontsize=FS_LABEL,
                          fontweight="bold")
        ax.set_title(ttl, fontsize=FS_TITLE, fontweight="bold")
        ax.grid(True, alpha=0.3)
        for ev_idx, ev_name in events:
            ax.axvline(x[ev_idx], color="purple", ls=":", alpha=0.6, lw=2.0)
        _bold_ticks(ax)
    # annotate events on the betti panel
    for ev_idx, ev_name in events:
        axes[1].annotate(EVENT_TO_PHASE.get(ev_name, ev_name),
                         (x[ev_idx], beta1[ev_idx]),
                         fontsize=FS_TICK - 2, color="purple", rotation=30,
                         fontweight="bold", ha="left", va="bottom")
    fig.suptitle("BRT Topology Evolution", fontsize=FS_SUPTITLE, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    _save(fig, out_dir, "topology_evolution")


def _save(fig, out_dir, stem):
    fig.savefig(join(out_dir, stem + ".jpg"), dpi=220, bbox_inches="tight")
    fig.savefig(join(out_dir, stem + ".pdf"), bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--scenario", choices=["origin", "ring", "spread"], default="ring")
    ap.add_argument("--n-predators", type=int, default=7)
    ap.add_argument("--ring-radius", type=float, default=1.6)
    ap.add_argument("--r-capture", type=float, default=0.55)
    ap.add_argument("--delta", type=float, default=0.08)
    ap.add_argument("--n-samples", type=int, default=600)
    ap.add_argument("--max-iters", type=int, default=6)
    ap.add_argument("--grid-res", type=int, default=96)
    ap.add_argument("--time-steps", type=int, default=16)
    ap.add_argument("--n-snapshots", type=int, default=6)
    ap.add_argument("--n-birds", type=int, default=8000)
    ap.add_argument("--n-flocks", type=int, default=6)
    ap.add_argument("--t-end", type=float, default=2.0)
    ap.add_argument("--extent", type=float, default=4.0)
    ap.add_argument("--smooth-sigma", type=float, default=1.3,
                    help="Gaussian sigma (grid px) to denoise MC speckle before topology")
    ap.add_argument("--label", type=str, default="")
    ap.add_argument("--iter-id", type=str, default="")
    ap.add_argument("--replot-cache", type=str, default="",
                    help="Load a saved plot cache (.pkl) and re-render figures only "
                         "(skips the solve). Use to iterate on figure styling cheaply.")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ---- fast path: re-render from a cached solve, no re-solving ----
    if args.replot_cache:
        import pickle
        with open(args.replot_cache, "rb") as fh:
            C = pickle.load(fh)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fig_brt_evolution(args.out_dir, C["snaps"], C["x_extent"],
                              C["centers"], C["r_capture"], C["label"])
            fig_brt_tube_3d(args.out_dir, C["snaps"], C["x_extent"], C["label"])
            fig_phase_space(args.out_dir, C["flock_states"], C["rep_snap"],
                            C["x_extent"], C["centers"], C["r_capture"], C["label"])
            fig_topology(args.out_dir, C["btimes_all"], C["chi"], C["beta1"],
                         C["ncomp"], C["events"], C["label"])
        print(f"Re-rendered figures from cache {args.replot_cache} -> {args.out_dir}")
        return
    x_extent = (-args.extent, args.extent, -args.extent, args.extent)
    centers = predator_centers(args.scenario, args.n_predators, args.ring_radius)
    label = args.label or f"{args.scenario} P{len(centers)} r={args.r_capture} d={args.delta}"

    cfg = SolverConfig(
        delta=args.delta, num_samples=args.n_samples,
        max_quasi_iters=args.max_iters, quasi_tol=1e-5,
        t_start=0.0, t_end=args.t_end, gradient_mode="b17",
        chunk_size=20000, n_predators=len(centers), n_flocks=args.n_flocks,
        time_steps=args.time_steps,
    )
    distributor = GPUDistributor(auto_detect=True)
    H = MurmuationHamiltonian4D(omega_e_bar=1.0, omega_p_bar=1.0, gamma_max=0.5)
    g = make_union_terminal_cost(centers, args.r_capture)
    sampler = HJReachabilitySampler(H, g, cfg, distributor)

    # backward-time grid: t from t_end down to 0; backward time tau = t_end - t
    t_vals = np.linspace(cfg.t_end, cfg.t_start, args.time_steps)

    topos, chi, beta1, ncomp, radii, grids, nan_fracs = [], [], [], [], [], [], []
    print(f"[{args.iter_id}] {label}: {args.time_steps} steps, grid {args.grid_res}")
    t0 = time.time()
    for i, t in enumerate(t_vals):
        V, X1, X2, nanf = build_value_grid(sampler, t, args.grid_res, x_extent,
                                           smooth_sigma=args.smooth_sigma)
        tp = brt_topology_signature(V)
        tp.time_idx = i
        topos.append(tp)
        chi.append(tp.euler_char); beta1.append(int(tp.betti_1))
        ncomp.append(tp.n_components); nan_fracs.append(nanf)
        # physical BRT radius (grid units -> meters)
        r_units = brt_radius_at_time(V)
        radii.append(r_units * (2 * args.extent / args.grid_res))
        grids.append((V, X1, X2))
        # Only surface the masked-cell fraction when it is non-trivial; a few
        # percent is normal float underflow far from the BRT, but a large value
        # signals --n-samples too low or --delta too small.
        warn = f"  [warn] {nanf:.1%} cells masked (NaN)" if nanf > NAN_WARN_FRAC else ""
        print(f"  step {i:2d} tau={cfg.t_end - t:.2f} chi={tp.euler_char:.0f} "
              f"b1={tp.betti_1} nc={tp.n_components}{warn}")
    wall = time.time() - t0

    events = detect_phase_transitions(topos, v_slices=[grd[0] for grd in grids])
    phases = label_phases(topos, radii)
    btimes_all = [cfg.t_end - t for t in t_vals]

    # choose snapshot indices (evenly spaced, skip the all-NaN terminal if empty)
    valid = [i for i in range(len(grids)) if ncomp[i] > 0] or list(range(len(grids)))
    snap_idx = np.linspace(valid[0], valid[-1], args.n_snapshots).round().astype(int)
    snap_idx = sorted(set(snap_idx.tolist()))
    snaps = []
    for i in snap_idx:
        V, X1, X2 = grids[i]
        snaps.append(dict(V=V, X1=X1, X2=X2, btime=btimes_all[i], phase=phases[i],
                          chi=chi[i], beta1=beta1[i], ncomp=ncomp[i]))

    # representative snapshot for phase-space = the one with richest topology
    rep_i = int(np.argmax([beta1[i] * 10 + ncomp[i] for i in range(len(grids))]))
    repV, repX1, repX2 = grids[rep_i]
    rep_snap = dict(V=repV, X1=repX1, X2=repX2, btime=btimes_all[rep_i])

    # flocks (for phase-space scatter)
    flocks = make_flocks(args.n_birds, args.n_flocks, x_extent)
    flock_states = np.concatenate([np.asarray(f.states) for f in flocks], 0)

    # safe fraction at tau_max (t=0): fraction of birds with v>0
    vb, _ = sampler.solve_quasi_linear(jnp.asarray(flock_states.astype(np.float32)), 0.0)
    vb = np.asarray(vb)
    safe_fraction = float(np.mean(vb[np.isfinite(vb)] > 0)) if np.isfinite(vb).any() else float("nan")

    # ---- save a plot cache so figures can be re-styled without re-solving ----
    import pickle
    cache = dict(snaps=snaps, rep_snap=rep_snap, flock_states=flock_states,
                 x_extent=x_extent, centers=centers, r_capture=args.r_capture,
                 label=label, btimes_all=btimes_all, chi=chi, beta1=beta1,
                 ncomp=ncomp, events=events)
    with open(join(args.out_dir, "plot_cache.pkl"), "wb") as fh:
        pickle.dump(cache, fh)

    # render
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fig_brt_evolution(args.out_dir, snaps, x_extent, centers, args.r_capture, label)
        tube_segs = fig_brt_tube_3d(args.out_dir, grids and snaps, x_extent, label)
        fig_phase_space(args.out_dir, flock_states, rep_snap, x_extent, centers,
                        args.r_capture, label)
        fig_topology(args.out_dir, btimes_all, chi, beta1, ncomp, events, label)

    meta = dict(
        iter_id=args.iter_id, label=label, scenario=args.scenario,
        n_birds=int(flock_states.shape[0]), n_flocks=args.n_flocks,
        n_predators=int(len(centers)), ring_radius=args.ring_radius,
        r_capture=args.r_capture, delta=args.delta, n_samples=args.n_samples,
        max_iters=args.max_iters, grid_res=args.grid_res, time_steps=args.time_steps,
        extent=args.extent, smooth_sigma=args.smooth_sigma,
        wall_clock_seconds=round(wall, 2),
        safe_fraction=safe_fraction, max_beta1=int(max(beta1)),
        min_chi=float(min(chi)), max_ncomp=int(max(ncomp)),
        events=[[int(i), n] for i, n in events],
        phases_per_step=phases, chi=chi, beta1=beta1, ncomp=ncomp,
        radii_m=[round(r, 3) for r in radii],
        nan_frac_per_step=[round(x, 4) for x in nan_fracs],
        centers=centers.tolist(),
    )
    with open(join(args.out_dir, "metadata.json"), "w") as fh:
        json.dump(meta, fh, indent=2)

    print(f"[{args.iter_id}] wall={wall:.1f}s safe={safe_fraction:.2%} "
          f"max_b1={max(beta1)} min_chi={min(chi):.0f} max_nc={max(ncomp)} "
          f"events={len(events)} tube_segs={tube_segs}")
    print(json.dumps({k: meta[k] for k in
          ["wall_clock_seconds", "safe_fraction", "max_beta1", "min_chi",
           "max_ncomp", "events"]}))


if __name__ == "__main__":
    main()
