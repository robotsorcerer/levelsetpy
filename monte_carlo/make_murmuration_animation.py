#!/usr/bin/env python
"""Animate the murmuration BRT: cordon -> vacuole -> cohesion, in backward time.

Reuses the solver wiring and rendering conventions of make_pub_figures.py (the
script that produced brt_evolution.jpg for the paper), but solves on a denser
time axis and emits an animation plus a static filmstrip.

Outputs (into --out-dir):
  murmuration_brt.gif        -- animated sweep in backward time tau
  murmuration_brt_strip.png  -- 6-panel static filmstrip (PDF fallback)

Needs a working JAX. Run with an env that has jax/scipy/matplotlib, e.g.:
    /tmp/murm_env/bin/python make_murmuration_animation.py --out-dir <assets>
"""
from __future__ import annotations
import os
import sys
import time
import argparse

_MC = os.path.dirname(os.path.abspath(__file__))
if _MC not in sys.path:
    sys.path.insert(0, _MC)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Circle
from PIL import Image

import make_pub_figures as P          # solver wiring + helpers + rc style
from src.config import SolverConfig
from src.gpu_distribution import GPUDistributor
from src.hj_sampler import HJReachabilitySampler
from src.hamiltonians.murmuration import MurmuationHamiltonian4D
from src.topology import brt_topology_signature, brt_radius_at_time

SENT = P.SENTINEL


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    # defaults = the paper's "actual usage" settings, denser in time
    ap.add_argument("--n-predators", type=int, default=7)
    ap.add_argument("--ring-radius", type=float, default=1.15)
    ap.add_argument("--r-capture", type=float, default=0.6)
    ap.add_argument("--delta", type=float, default=0.18)
    ap.add_argument("--n-samples", type=int, default=1600)
    ap.add_argument("--max-iters", type=int, default=7)
    ap.add_argument("--grid-res", type=int, default=112)
    ap.add_argument("--time-steps", type=int, default=24)
    ap.add_argument("--n-flocks", type=int, default=6)
    ap.add_argument("--t-end", type=float, default=2.0)
    ap.add_argument("--extent", type=float, default=3.5)
    ap.add_argument("--smooth-sigma", type=float, default=1.5)
    ap.add_argument("--fps", type=int, default=5)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    x_extent = (-args.extent, args.extent, -args.extent, args.extent)
    centers = P.predator_centers("ring", args.n_predators, args.ring_radius)

    cfg = SolverConfig(delta=args.delta, num_samples=args.n_samples,
                       max_quasi_iters=args.max_iters, quasi_tol=1e-5,
                       t_start=0.0, t_end=args.t_end, gradient_mode="b17",
                       chunk_size=20000, n_predators=len(centers),
                       n_flocks=args.n_flocks, time_steps=args.time_steps)
    H = MurmuationHamiltonian4D(omega_e_bar=1.0, omega_p_bar=1.0, gamma_max=0.5)
    g = P.make_union_terminal_cost(centers, args.r_capture)
    sampler = HJReachabilitySampler(H, g, cfg, GPUDistributor(auto_detect=True))

    t_vals = np.linspace(cfg.t_end, cfg.t_start, args.time_steps)
    grids, topos, radii = [], [], []
    t0 = time.time()
    for i, t in enumerate(t_vals):
        V, X1, X2, nanf = P.build_value_grid(sampler, t, args.grid_res, x_extent,
                                             smooth_sigma=args.smooth_sigma)
        tp = brt_topology_signature(V)
        grids.append((V, X1, X2)); topos.append(tp)
        radii.append(brt_radius_at_time(V) * (2 * args.extent / args.grid_res))
        print(f"  step {i:2d} tau={cfg.t_end - t:.2f} chi={tp.euler_char:.0f} "
              f"b1={tp.betti_1} nc={tp.n_components} masked={nanf:.1%}")
    print(f"solve wall-clock: {time.time() - t0:.1f}s")

    phases = P.label_phases(topos, radii)
    btimes = [cfg.t_end - t for t in t_vals]

    finite = np.concatenate([V[np.isfinite(V) & (V < SENT / 2)].ravel()
                             for V, _, _ in grids])
    vmax = max(float(np.nanpercentile(np.abs(finite), 98)) if finite.size else 1.0, 0.5)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    def draw(ax, i, fs=1.0):
        V, X1, X2 = grids[i]
        ax.pcolormesh(X1, X2, np.where(V >= SENT / 2, vmax, V), cmap="RdBu_r",
                      norm=norm, shading="auto", rasterized=True)
        ax.contourf(X1, X2, V, levels=[-1e9, 0.0], colors=["#2b2b6f"], alpha=0.30)
        if np.nanmin(V) <= 0 <= np.nanmax(V):
            ax.contour(X1, X2, V, levels=[0.0], colors="k", linewidths=3.0)
        for c in centers:
            ax.add_patch(Circle((c[0], c[1]), args.r_capture, fill=False,
                                ec="lime", lw=2.0, ls="--", alpha=0.9))
            ax.plot(c[0], c[1], marker="x", color="lime", ms=9, mew=2.5)
        ax.set_xlim(-args.extent, args.extent)
        ax.set_ylim(-args.extent, args.extent)
        ax.set_aspect("equal")
        tp = topos[i]
        ax.set_title(f"$\\tau$={btimes[i]:.2f}  [{phases[i]}]\n"
                     f"$\\chi$={tp.euler_char:.0f}  $\\beta_1$={tp.betti_1}  "
                     f"$n_c$={tp.n_components}",
                     fontsize=int(20 * fs), fontweight="bold")
        ax.set_xlabel("$x_1$ (m)", fontsize=int(18 * fs), fontweight="bold")
        ax.set_ylabel("$x_2$ (m)", fontsize=int(18 * fs), fontweight="bold")
        ax.tick_params(labelsize=int(13 * fs))

    frames = []
    for i in range(len(grids)):
        fig, ax = plt.subplots(figsize=(6.0, 6.2), dpi=84)
        draw(ax, i, fs=0.72)
        sm = cm.ScalarMappable(norm=norm, cmap="RdBu_r"); sm.set_array([])
        cb = fig.colorbar(sm, ax=ax, shrink=0.82, pad=0.02)
        cb.set_label("value $v(x)$", fontsize=12, fontweight="bold")
        cb.ax.tick_params(labelsize=10)
        fig.subplots_adjust(left=0.13, right=0.99, top=0.88, bottom=0.10)
        fig.canvas.draw()
        frames.append(Image.fromarray(
            np.asarray(fig.canvas.buffer_rgba())[..., :3]).convert(
                "P", palette=Image.ADAPTIVE, colors=128))
        plt.close(fig)

    frames = frames + frames[::-1][1:]      # sweep and rewind: clean loop
    gif = os.path.join(args.out_dir, "murmuration_brt.gif")
    frames[0].save(gif, save_all=True, append_images=frames[1:], loop=0,
                   duration=int(1000 / args.fps), optimize=True)
    print(f"wrote {gif} ({len(frames)} frames, "
          f"{os.path.getsize(gif) / 1e6:.1f} MB)")

    picks = np.linspace(0, len(grids) - 1, 6).round().astype(int)
    fig, axes = plt.subplots(2, 3, figsize=(15.0, 10.2), dpi=100,
                             gridspec_kw=dict(wspace=0.26, hspace=0.34))
    for a, i in zip(axes.ravel(), picks):
        draw(a, int(i), fs=0.9)
    sm = cm.ScalarMappable(norm=norm, cmap="RdBu_r"); sm.set_array([])
    cb = fig.colorbar(sm, ax=axes.ravel().tolist(), shrink=0.85, pad=0.015)
    cb.set_label("value $v(x)$", fontsize=15, fontweight="bold")
    strip = os.path.join(args.out_dir, "murmuration_brt_strip.png")
    fig.savefig(strip, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {strip}")

    # topology figure from THIS run, so the animation and the chi/beta plot agree
    chi = [tp.euler_char for tp in topos]
    beta1 = [int(tp.betti_1) for tp in topos]
    ncomp = [tp.n_components for tp in topos]
    try:
        from src.topology import detect_phase_transitions
        events = detect_phase_transitions(topos, v_slices=[g[0] for g in grids])
    except Exception as exc:                                   # non-fatal
        print("  (phase-transition detection skipped:", exc, ")")
        events = []
    P.fig_topology(args.out_dir, btimes, chi, beta1, ncomp, events,
                   "murmuration ring cordon")
    print("wrote", os.path.join(args.out_dir, "topology_evolution.jpg"))

    # report the topological event times for the slide text
    for i in range(1, len(topos)):
        if (topos[i].betti_1 != topos[i - 1].betti_1
                or topos[i].n_components != topos[i - 1].n_components):
            print(f"  event at tau={btimes[i]:.2f}: "
                  f"b1 {topos[i-1].betti_1}->{topos[i].betti_1}, "
                  f"nc {topos[i-1].n_components}->{topos[i].n_components}, "
                  f"phase {phases[i-1]}->{phases[i]}")


if __name__ == "__main__":
    main()
