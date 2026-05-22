#!/usr/bin/env python
"""Double integrator backward reachable set via MC Cole-Hopf.

State: (position, velocity), control: acceleration with |u| <= 1.
Target set: sphere of radius 0.5 at the origin.

Produces: examples/double_integrator.png
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from functools import partial

from src.config import SolverConfig
from src.hamiltonians import DoubleIntegratorHamiltonian
from src.initial_conditions import sphere_cost
from src.hj_sampler import HJReachabilitySampler

# ── Parameters ──────────────────────────────────────────────────────
U_BOUND = 1.0
RADIUS = 0.5
GRID_RES = 40          # per axis
DOMAIN = (-3.0, 3.0)
TIMES = [0.8, 0.5, 0.2, 0.0]   # backward from T=1

cfg = SolverConfig(
    delta=0.08,
    num_samples=8_000,
    max_quasi_iters=15,
    quasi_tol=1e-5,
    t_start=0.0,
    t_end=1.0,
    seed=0,
)

# ── Build grid ──────────────────────────────────────────────────────
xs = jnp.linspace(*DOMAIN, GRID_RES)
X1, X2 = jnp.meshgrid(xs, xs, indexing="ij")
eval_points = jnp.stack([X1.ravel(), X2.ravel()], axis=-1)  # (M, 2)

# ── Hamiltonian and terminal cost ───────────────────────────────────
H = DoubleIntegratorHamiltonian(u_bound=U_BOUND)
g = partial(sphere_cost, radius=RADIUS)

# ── Solve and plot ──────────────────────────────────────────────────
fig, axes = plt.subplots(1, len(TIMES), figsize=(4 * len(TIMES), 4),
                         sharex=True, sharey=True)

for ax, t in zip(axes, TIMES):
    print(f"Solving at t = {t:.2f} ...")
    sampler = HJReachabilitySampler(H, g, cfg)
    v, history = sampler.solve_quasi_linear(eval_points, t)
    V = v.reshape(GRID_RES, GRID_RES)

    ax.contourf(X1, X2, V, levels=20, cmap="RdBu_r")
    ax.contour(X1, X2, V, levels=[0.0], colors="k", linewidths=2)
    ax.set_title(f"t = {t:.2f}  ({len(history)} iters)")
    ax.set_xlabel("position")
    ax.set_ylabel("velocity")
    ax.set_aspect("equal")

fig.suptitle("Double Integrator BRS (zero level set = boundary)", fontsize=13)
fig.tight_layout()
out = os.path.join(os.path.dirname(__file__), "double_integrator.png")
fig.savefig(out, dpi=150)
print(f"Saved → {out}")
plt.close(fig)
