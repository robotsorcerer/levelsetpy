#!/usr/bin/env python
"""Dubins vehicle backward reachable set via MC Cole-Hopf.

State: (x, y, theta), control: turn rate |u| <= omega_max.
Target set: sphere of radius 0.5 at the origin in (x, y, theta).

We fix theta at several values and plot 2-D (x, y) slices of the
zero level set at t = 0.

Produces: examples/dubins.png
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from functools import partial
from math import pi

from src.config import SolverConfig
from src.hamiltonians import DubinsHamiltonian
from src.initial_conditions import sphere_cost
from src.hj_sampler import HJReachabilitySampler

# ── Parameters ──────────────────────────────────────────────────────
SPEED = 1.0
OMEGA_MAX = 1.0
RADIUS = 0.5
GRID_RES = 35
DOMAIN = (-3.0, 3.0)
THETA_SLICES = [-pi / 2, 0.0, pi / 4, pi / 2]
T_EVAL = 0.0       # evaluate at t=0 (full backward horizon)

cfg = SolverConfig(
    delta=0.08,
    num_samples=8_000,
    max_quasi_iters=15,
    quasi_tol=1e-5,
    t_start=0.0,
    t_end=1.0,
    seed=1,
)

# ── Hamiltonian and terminal cost ───────────────────────────────────
H = DubinsHamiltonian(speed=SPEED, omega_max=OMEGA_MAX)
g = partial(sphere_cost, radius=RADIUS)

# ── Solve at each theta slice ───────────────────────────────────────
xs = jnp.linspace(*DOMAIN, GRID_RES)
X, Y = jnp.meshgrid(xs, xs, indexing="ij")

fig, axes = plt.subplots(1, len(THETA_SLICES),
                         figsize=(4 * len(THETA_SLICES), 4),
                         sharex=True, sharey=True)

for ax, theta_val in zip(axes, THETA_SLICES):
    # Build 3-D eval points with fixed theta
    theta_col = jnp.full((GRID_RES * GRID_RES, 1), theta_val)
    eval_points = jnp.concatenate(
        [X.ravel()[:, None], Y.ravel()[:, None], theta_col], axis=-1
    )

    print(f"Solving theta = {theta_val:.2f} ...")
    sampler = HJReachabilitySampler(H, g, cfg)
    v, history = sampler.solve_quasi_linear(eval_points, T_EVAL)
    V = v.reshape(GRID_RES, GRID_RES)

    ax.contourf(X, Y, V, levels=20, cmap="RdBu_r")
    ax.contour(X, Y, V, levels=[0.0], colors="k", linewidths=2)
    ax.set_title(rf"$\theta = {theta_val:.2f}$  ({len(history)} iters)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")

fig.suptitle("Dubins Vehicle BRS slices (zero level set)", fontsize=13)
fig.tight_layout()
out = os.path.join(os.path.dirname(__file__), "dubins.png")
fig.savefig(out, dpi=150)
print(f"Saved → {out}")
plt.close(fig)
