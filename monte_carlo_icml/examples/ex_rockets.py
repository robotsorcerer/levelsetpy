#!/usr/bin/env python
"""Two-rockets pursuit-evasion reachable set via MC Cole-Hopf.

State: (x, z, theta) in relative coordinates.
  - (x, z): relative position of evader w.r.t. pursuer
  - theta: relative thrust inclination

Target set: cylinder of radius 1.5 (infinite along theta).

We fix theta at several values and plot 2-D (x, z) cross-sections
of the backward reachable tube at t = 0.

Produces: examples/rockets.png
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from functools import partial
from math import pi

from src.config import SolverConfig
from src.hamiltonians import RocketsRelativeHamiltonian
from src.initial_conditions import cylinder_cost
from src.hj_sampler import HJReachabilitySampler

# ── Parameters ──────────────────────────────────────────────────────
A = 1.0          # thrust magnitude
G = 32.0         # gravity
U_BOUND = 1.0    # max turn rate
RADIUS = 1.5     # cylinder radius
GRID_RES = 35
DOMAIN = (-5.0, 5.0)
THETA_SLICES = [-pi / 4, 0.0, pi / 4, pi / 2]
T_EVAL = 0.0

cfg = SolverConfig(
    delta=0.08,
    num_samples=8_000,
    max_quasi_iters=15,
    quasi_tol=1e-5,
    t_start=0.0,
    t_end=0.5,
    seed=2,
)

# ── Hamiltonian and terminal cost ───────────────────────────────────
H = RocketsRelativeHamiltonian(a=A, g=G, u_bound=U_BOUND)
g = partial(cylinder_cost, axis_align=2, radius=RADIUS)

# ── Solve at each theta slice ───────────────────────────────────────
xs = jnp.linspace(*DOMAIN, GRID_RES)
X, Z = jnp.meshgrid(xs, xs, indexing="ij")

fig, axes = plt.subplots(1, len(THETA_SLICES),
                         figsize=(4 * len(THETA_SLICES), 4),
                         sharex=True, sharey=True)

for ax, theta_val in zip(axes, THETA_SLICES):
    theta_col = jnp.full((GRID_RES * GRID_RES, 1), theta_val)
    eval_points = jnp.concatenate(
        [X.ravel()[:, None], Z.ravel()[:, None], theta_col], axis=-1
    )

    print(f"Solving theta = {theta_val:.2f} ...")
    sampler = HJReachabilitySampler(H, g, cfg)
    v, history = sampler.solve_quasi_linear(eval_points, T_EVAL)
    V = v.reshape(GRID_RES, GRID_RES)

    ax.contourf(X, Z, V, levels=20, cmap="RdBu_r")
    ax.contour(X, Z, V, levels=[0.0], colors="k", linewidths=2)
    ax.set_title(rf"$\theta = {theta_val:.2f}$  ({len(history)} iters)")
    ax.set_xlabel("x (relative)")
    ax.set_ylabel("z (relative)")
    ax.set_aspect("equal")

fig.suptitle("Two-Rockets Pursuit-Evasion BRT slices", fontsize=13)
fig.tight_layout()
out = os.path.join(os.path.dirname(__file__), "rockets.png")
fig.savefig(out, dpi=150)
print(f"Saved → {out}")
plt.close(fig)
