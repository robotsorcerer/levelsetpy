#!/usr/bin/env python
"""Two-Rockets Pursuit-Evasion: JAX Implementation with JIT Compilation.

Implements the quasi-linearization algorithm (Algorithm 1) from:

    "Approximately Correct and Scalable HJ-Reachability: A Sampling Scheme"
    ICML 2026

for the rocket launch problem described in Section 4.1.

This module mirrors ``rocket_python.py`` but uses JAX for:
  - JIT compilation of hot loops
  - Vectorized operations via vmap
  - Automatic differentiation for gradients
  - GPU/TPU acceleration (when available)

Problem Setup: See rocket_python.py docstring or Section 4.1 of the paper.
"""

import jax
import jax.numpy as jnp
from jax import vmap, jit
from functools import partial
from math import pi
from typing import Tuple, List, Optional, NamedTuple
import numpy as np
import time as time_mod
from pathlib import Path

from config_loaders import _load_rocket_config
# ============================================================================
#  Configuration (JAX-compatible NamedTuple)
# ============================================================================

class SolverConfig(NamedTuple):
    """Immutable solver configuration (JAX PyTree-compatible).

    All fields are scalars so this is a valid JAX static argument.
    """
    delta: float = 0.08
    num_samples: int = 10_000
    max_iters: int = 20
    tol: float = 1e-5
    t_start: float = 0.0
    t_end: float = 1.0
    time_steps: int = 50
    seed: int = 123
    smoothing_eps: float = 1e-4


# Load config as NamedTuple
CONFIG = _load_rocket_config()

# ============================================================================
#  Hamiltonian (Eq 22-23) -- JAX version
# ============================================================================
def rockets_hamiltonian(
    t: float,
    x: jnp.ndarray,
    p: jnp.ndarray,
    a: float = CONFIG.A_THRUST,
    g: float = CONFIG.GRAVITY,
    u_bound: float = CONFIG.U_BOUND,
    eps: float = 1e-4,
) -> jnp.ndarray:
    r"""Two-rockets Hamiltonian H(t, x, p) in relative coordinates.

    Eq (23) of the paper:
        H = -a p1 cos(theta) - p2(g - a - a sin(theta))
            + u_bar |p2*x + p3| - u_bar |p1*x + p3|

    Using smooth absolute value |z| ~ sqrt(z^2 + eps^2) for
    differentiability (required by JAX autodiff).
    """
    p1, p2, p3 = p[0], p[1], p[2]
    x_pos = x[0]
    theta = x[2]

    smooth_abs = lambda z: jnp.sqrt(z ** 2 + eps ** 2)

    term1 = -a * p1 * jnp.cos(theta)
    term2 = -p2 * (g - a - a * jnp.sin(theta))
    term3 = u_bound * smooth_abs(p2 * x_pos + p3)
    term4 = -u_bound * smooth_abs(p1 * x_pos + p3)

    return term1 + term2 + term3 + term4


# ============================================================================
#  Terminal cost (signed distance to capture cylinder)
# ============================================================================

def rockets_terminal_cost(
    x: jnp.ndarray,
    radius: float = CONFIG.CAPTURE_RADIUS,
) -> jnp.ndarray:
    r"""g(x) = sqrt(x^2 + z^2) - ell.

    Cylinder cost: ignore theta dimension (axis_align=2).
    """
    spatial = x[:2]  # (x_pos, z_pos)
    return jnp.linalg.norm(spatial) - radius


# ============================================================================
#  MC value function (Eq 18) -- JIT-compiled
# ============================================================================

@partial(jit, static_argnums=(6,))
def mc_value_at_point(
    key: jax.Array,
    x: jnp.ndarray,
    t: float,
    T: float,
    delta: float,
    c: float,
    num_samples: int,
) -> jnp.ndarray:
    r"""Compute v^delta(t; x) via MC Gaussian expectation.

    Eq (18): v(t;x) = -(1/c) log (1/N) sum exp(-c g(x + sigma*z_i))

    where sigma = sqrt(delta*(T-t)), z_i ~ N(0, I).
    """
    n = x.shape[0]
    sigma = jnp.sqrt(jnp.maximum(delta * (T - t), 1e-30))

    z = jax.random.normal(key, shape=(num_samples, n))
    y = x[None, :] + sigma * z  # (N, n)

    g_vals = vmap(rockets_terminal_cost)(y)  # (N,)

    # Stable log-sum-exp
    exponents = -c * g_vals
    max_exp = jnp.max(exponents)
    log_mean_exp = max_exp + jnp.log(jnp.mean(jnp.exp(exponents - max_exp)))

    return -(1.0 / c) * log_mean_exp


@partial(jit, static_argnums=(6,))
def mc_gradient_at_point(
    key: jax.Array,
    x: jnp.ndarray,
    t: float,
    T: float,
    delta: float,
    c: float,
    num_samples: int,
) -> jnp.ndarray:
    r"""Compute Dv^delta(t; x) via importance-weighted MC.

    Eq (19) / Corollary 3.7:
        Dv = (1/(t_eff * delta * c)) * (x - sum w_i d_i / sum w_i)

    where d_i = x + sigma * z_i, w_i = exp(-c g(d_i)).
    """
    n = x.shape[0]
    sigma = jnp.sqrt(jnp.maximum(delta * (T - t), 1e-30))
    t_eff = jnp.maximum(T - t, 1e-30)

    z = jax.random.normal(key, shape=(num_samples, n))
    d = x[None, :] + sigma * z  # (N, n)

    g_vals = vmap(rockets_terminal_cost)(d)  # (N,)

    # Stabilized importance weights
    log_w = -c * g_vals
    log_w_shifted = log_w - jnp.max(log_w)
    weights = jnp.exp(log_w_shifted)
    weights = weights / jnp.sum(weights)

    weighted_mean = jnp.sum(weights[:, None] * d, axis=0)

    return (1.0 / (t_eff * delta * c)) * (x - weighted_mean)


# ============================================================================
#  Batched MC solvers (vmap over evaluation points)
# ============================================================================

def mc_value_batch(
    key: jax.Array,
    eval_points: jnp.ndarray,
    t: float,
    T: float,
    delta: float,
    c: jnp.ndarray,
    num_samples: int,
) -> jnp.ndarray:
    """Evaluate v(t; .) at M points via vmap."""
    M = eval_points.shape[0]
    keys = jax.random.split(key, M)
    c_arr = jnp.broadcast_to(jnp.asarray(c, dtype=jnp.float32), (M,))

    def _solve_one(k, xi, ci):
        return mc_value_at_point(k, xi, t, T, delta, ci, num_samples)

    return vmap(_solve_one)(keys, eval_points, c_arr)


def mc_gradient_batch(
    key: jax.Array,
    eval_points: jnp.ndarray,
    t: float,
    T: float,
    delta: float,
    c: jnp.ndarray,
    num_samples: int,
) -> jnp.ndarray:
    """Evaluate Dv(t; .) at M points via vmap."""
    M = eval_points.shape[0]
    keys = jax.random.split(key, M)
    c_arr = jnp.broadcast_to(jnp.asarray(c, dtype=jnp.float32), (M,))

    def _grad_one(k, xi, ci):
        return mc_gradient_at_point(k, xi, t, T, delta, ci, num_samples)

    return vmap(_grad_one)(keys, eval_points, c_arr)


# ============================================================================
#  Frozen coefficient utilities
# ============================================================================

@jit
def compute_frozen_coefficient(
    H_vals: jnp.ndarray,
    grad_v_sq: jnp.ndarray,
    delta: float,
    eps: float = 1e-8,
) -> jnp.ndarray:
    """c(t, x) = 2 H / (delta |Dv|^2)."""
    return 2.0 * H_vals / (delta * jnp.maximum(grad_v_sq, eps))


@jit
def relative_residual(v_new: jnp.ndarray, v_old: jnp.ndarray) -> jnp.ndarray:
    """||v_new - v_old|| / ||v_old||."""
    return jnp.linalg.norm(v_new - v_old) / jnp.maximum(
        jnp.linalg.norm(v_old), 1e-12
    )


# ============================================================================
#  Quasi-Linearization Solver (Algorithm 1) -- JAX version
# ============================================================================

def solve_quasi_linear_jax(
    eval_points: jnp.ndarray,
    t: float,
    T: float,
    delta: float,
    c_init: jnp.ndarray,
    num_samples: int,
    max_iters: int,
    tol: float,
    key: jax.Array,
) -> Tuple[jnp.ndarray, List[float]]:
    r"""Quasi-linearization iteration (Algorithm 1) using JAX.

    This is the JAX-accelerated version of Algorithm 1. Each iteration:
      1. Freeze c^{(k)}
      2. Solve heat equation via MC Gaussian (Eq 18)
      3. Recover v^{(k+1)} = -(1/c^{(k)}) log omega^{(k)}
      4. Update Dv^{(k+1)} (Eq 19) and c^{(k+1)}
      5. Check convergence

    Parameters
    ----------
    eval_points : jnp.ndarray, shape (M, n)
    t : float, current time
    T : float, terminal time
    delta : float, viscosity
    c_init : jnp.ndarray, shape (M,), initial frozen coefficients
    num_samples : int
    max_iters : int
    tol : float
    key : JAX PRNG key

    Returns
    -------
    v : jnp.ndarray, shape (M,)
    history : list of float
    """
    M = eval_points.shape[0]

    # Initialize: v^{(0)}(t;x) = g(x)
    v_current = vmap(rockets_terminal_cost)(eval_points)
    c_frozen = jnp.abs(c_init) + 1e-8

    history: List[float] = []

    for k in range(max_iters):
        # Steps 2-3: MC value function with frozen c
        key, subkey = jax.random.split(key)
        v_new = mc_value_batch(
            subkey, eval_points, t, T, delta, c_frozen, num_samples,
        )

        # Step 5: convergence
        resid = float(relative_residual(v_new, v_current))
        history.append(resid)
        v_current = v_new

        if resid < tol:
            break

        # Step 4: Update gradient and c
        key, subkey = jax.random.split(key)
        Dv = mc_gradient_batch(
            subkey, eval_points, t, T, delta, c_frozen, num_samples,
        )

        # Hamiltonian at (t, x, Dv)
        H_vals = vmap(
            lambda x, p: rockets_hamiltonian(t, x, p)
        )(eval_points, Dv)

        grad_v_sq = jnp.sum(Dv ** 2, axis=-1)
        c_frozen = compute_frozen_coefficient(H_vals, grad_v_sq, delta)
        c_frozen = jnp.abs(c_frozen) + 1e-8

    return v_current, history


# ============================================================================
#  High-level solver class
# ============================================================================

class RocketsSolverJAX:
    """JAX-accelerated solver for the two-rockets problem.

    Uses JIT compilation and vmap for efficient batch evaluation.
    """

    def __init__(self, config: Optional[SolverConfig] = None):
        self.cfg = config or SolverConfig(
            delta=0.08,
            num_samples=8_000,
            max_iters=15,
            tol=1e-5,
            t_start=0.0,
            t_end=0.5,
            seed=42,
        )
        self.key = jax.random.PRNGKey(self.cfg.seed)

    def _compute_initial_c(
        self, eval_points: jnp.ndarray, t: float,
    ) -> jnp.ndarray:
        """Compute c^{(0)} = 2 H(t;x,Dg) / (delta |Dg|^2)."""
        # Dg via JAX autodiff
        Dg_fn = jax.grad(rockets_terminal_cost)
        Dg = vmap(Dg_fn)(eval_points)  # (M, n)

        # H(t, x, Dg)
        H_vals = vmap(
            lambda x, p: rockets_hamiltonian(t, x, p)
        )(eval_points, Dg)

        Dg_sq = jnp.sum(Dg ** 2, axis=-1)
        c_init = compute_frozen_coefficient(H_vals, Dg_sq, self.cfg.delta)
        return c_init

    def solve_slice(
        self,
        theta_val: float,
        grid_res: int = 40,
        domain: Tuple[float, float] = (-5.0, 5.0),
        t_eval: float = 0.0,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, List[float]]:
        """Solve on a 2D (x, z) slice at fixed theta."""
        xs = jnp.linspace(domain[0], domain[1], grid_res)
        X, Z = jnp.meshgrid(xs, xs, indexing="ij")

        theta_col = jnp.full((grid_res * grid_res, 1), theta_val)
        eval_points = jnp.concatenate(
            [X.ravel()[:, None], Z.ravel()[:, None], theta_col],
            axis=-1,
        )

        c_init = self._compute_initial_c(eval_points, t_eval)

        self.key, subkey = jax.random.split(self.key)
        v, history = solve_quasi_linear_jax(
            eval_points=eval_points,
            t=t_eval,
            T=self.cfg.t_end,
            delta=self.cfg.delta,
            c_init=c_init,
            num_samples=self.cfg.num_samples,
            max_iters=self.cfg.max_iters,
            tol=self.cfg.tol,
            key=subkey,
        )

        V = v.reshape(grid_res, grid_res)
        return X, Z, V, history

    def solve_volume(
        self,
        grid_res: int = 25,
        spatial_domain: Tuple[float, float] = (-5.0, 5.0),
        theta_range: Tuple[float, float] = (-pi / 2, pi / 2),
        t_eval: float = 0.0,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, List[float]]:
        """Solve on the full 3D grid."""
        xs = jnp.linspace(spatial_domain[0], spatial_domain[1], grid_res)
        ts = jnp.linspace(theta_range[0], theta_range[1], grid_res)
        XX, ZZ, TT = jnp.meshgrid(xs, xs, ts, indexing="ij")

        eval_points = jnp.stack(
            [XX.ravel(), ZZ.ravel(), TT.ravel()], axis=-1,
        )

        c_init = self._compute_initial_c(eval_points, t_eval)

        self.key, subkey = jax.random.split(self.key)
        v, history = solve_quasi_linear_jax(
            eval_points=eval_points,
            t=t_eval,
            T=self.cfg.t_end,
            delta=self.cfg.delta,
            c_init=c_init,
            num_samples=self.cfg.num_samples,
            max_iters=self.cfg.max_iters,
            tol=self.cfg.tol,
            key=subkey,
        )

        V = v.reshape(grid_res, grid_res, grid_res)
        return XX, ZZ, TT, V, history


# ============================================================================
#  Plotting
# ============================================================================

def plot_2d_slices(
    solver: RocketsSolverJAX,
    theta_slices: Optional[List[float]] = None,
    grid_res: int = 40,
    domain: Tuple[float, float] = (-5.0, 5.0),
    save_path: Optional[str] = None,
) -> None:
    """Plot 2D slices of the rockets BRT."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if theta_slices is None:
        theta_slices = [-pi / 4, 0.0, pi / 4, pi / 2]

    n_slices = len(theta_slices)
    fig, axes = plt.subplots(
        1, n_slices, figsize=(4 * n_slices, 4),
        sharex=True, sharey=True,
    )
    if n_slices == 1:
        axes = [axes]

    for ax, theta_val in zip(axes, theta_slices):
        print(f"  [JAX] Solving theta = {theta_val:.3f} ...")
        t0 = time_mod.time()
        X, Z, V, history = solver.solve_slice(
            theta_val, grid_res=grid_res, domain=domain,
        )
        elapsed = time_mod.time() - t0
        print(f"    {len(history)} iters, {elapsed:.1f}s")

        X_np, Z_np, V_np = np.array(X), np.array(Z), np.array(V)
        ax.contourf(X_np, Z_np, V_np, levels=20, cmap="RdBu_r")
        ax.contour(X_np, Z_np, V_np, levels=[0.0], colors="k", linewidths=2)
        ax.set_title(
            rf"$\theta = {theta_val:.2f}$  ({len(history)} iters)",
            fontsize=11, fontweight="bold",
        )
        ax.set_xlabel(r"$x$ (ft)", fontsize=11, fontweight="bold")
        ax.set_ylabel(r"$z$ (ft)", fontsize=11, fontweight="bold")
        ax.set_aspect("equal")

    fig.suptitle(
        "Two-Rockets BRT: Quasi-Linearized Cole-Hopf (JAX)",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved -> {save_path}")
    plt.close(fig)


# ============================================================================
#  Main
# ============================================================================

if __name__ == "__main__":
    import os

    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)

    cfg = SolverConfig(
        delta=0.08,
        num_samples=8_000,
        max_iters=20,
        tol=1e-5,
        t_start=0.0,
        t_end=0.5,
        seed=123,
    )

    solver = RocketsSolverJAX(config=cfg)

    print("=" * 60)
    print("Two-Rockets Pursuit-Evasion BRT (JAX)")
    print("=" * 60)

    t0 = time_mod.time()
    plot_2d_slices(
        solver,
        theta_slices=[-pi / 4, 0.0, pi / 4],
        grid_res=35,
        domain=(-5.0, 5.0),
        save_path=os.path.join(results_dir, "rocket_2d_slices.png"),
    )
    elapsed = time_mod.time() - t0
    print(f"Total time (JAX): {elapsed:.1f}s")
