#!/usr/bin/env python
"""3D Dubins Car Backward Reachability: JAX Implementation.

Implements the quasi-linearization algorithm (Algorithm 1) from:

    "Approximately Correct and Scalable HJ-Reachability: A Sampling Scheme"
    ICML 2026

for a 3D Dubins vehicle model using JAX for JIT compilation and vmap.

Supports both:
  - Simple Dubins car (single player)
  - Dubins pursuit-evasion (Merz 1972, two identical vehicles)

See dubins_python.py for detailed problem description.
"""

import jax
import jax.numpy as jnp
from jax import vmap, jit
from functools import partial
from math import pi
from typing import Tuple, List, Optional, NamedTuple
import numpy as np
import time as time_mod


# ============================================================================
#  Configuration
# ============================================================================

class SolverConfig(NamedTuple):
    """Immutable solver config (JAX PyTree-compatible)."""
    delta: float = 0.08
    num_samples: int = 10_000
    max_iters: int = 20
    tol: float = 1e-5
    t_start: float = 0.0
    t_end: float = 1.0
    time_steps: int = 50
    seed: int = 42
    smoothing_eps: float = 1e-4


# ============================================================================
#  Hamiltonians -- JAX versions
# ============================================================================

def dubins_simple_hamiltonian(
    t: float,
    x: jnp.ndarray,
    p: jnp.ndarray,
    speed: float = 1.0,
    omega_max: float = 1.0,
    eps: float = 1e-4,
) -> jnp.ndarray:
    """Simple Dubins car Hamiltonian.

    H(x, p) = v cos(theta) p1 + v sin(theta) p2 - omega_max |p3|
    """
    p1, p2, p3 = p[0], p[1], p[2]
    theta = x[2]
    abs_p3 = jnp.sqrt(p3 ** 2 + eps ** 2)
    return (speed * jnp.cos(theta) * p1
            + speed * jnp.sin(theta) * p2
            - omega_max * abs_p3)


def dubins_relative_hamiltonian(
    t: float,
    x: jnp.ndarray,
    p: jnp.ndarray,
    v_p: float = -1.0,
    v_e: float = 1.0,
    w: float = 1.0,
    eps: float = 1e-4,
) -> jnp.ndarray:
    """Dubins pursuit-evasion Hamiltonian (Merz 1972).

    H(x, p) = p1(v_e - v_p cos(x3)) - p2(v_p sin(x3))
              - w|p1 x2 - p2 x1 - p3| + w|p3|
    """
    p1, p2, p3 = p[0], p[1], p[2]
    x1, x2, x3 = x[0], x[1], x[2]

    smooth_abs = lambda z: jnp.sqrt(z ** 2 + eps ** 2)

    term1 = p1 * (v_e - v_p * jnp.cos(x3))
    term2 = -p2 * (v_p * jnp.sin(x3))
    term3 = -w * smooth_abs(p1 * x2 - p2 * x1 - p3)
    term4 = w * smooth_abs(p3)
    return term1 + term2 + term3 + term4


# ============================================================================
#  Terminal cost
# ============================================================================

def dubins_terminal_cost(
    x: jnp.ndarray,
    radius: float = 0.5,
) -> jnp.ndarray:
    """g(x) = ||(x, y)|| - radius (cylinder, ignore theta)."""
    spatial = x[:2]
    return jnp.linalg.norm(spatial) - radius


# ============================================================================
#  MC value and gradient -- JIT-compiled
# ============================================================================

def _make_mc_value_fn(terminal_cost_fn):
    """Create a JIT-compiled MC value function for a given terminal cost."""

    @partial(jit, static_argnums=(6,))
    def mc_value_at_point(
        key, x, t, T, delta, c, num_samples,
    ):
        n = x.shape[0]
        sigma = jnp.sqrt(jnp.maximum(delta * (T - t), 1e-30))
        z = jax.random.normal(key, shape=(num_samples, n))
        y = x[None, :] + sigma * z
        g_vals = vmap(terminal_cost_fn)(y)
        exponents = -c * g_vals
        max_exp = jnp.max(exponents)
        log_mean_exp = max_exp + jnp.log(jnp.mean(jnp.exp(exponents - max_exp)))
        return -(1.0 / c) * log_mean_exp

    return mc_value_at_point


def _make_mc_gradient_fn(terminal_cost_fn):
    """Create a JIT-compiled MC gradient function for a given terminal cost."""

    @partial(jit, static_argnums=(6,))
    def mc_gradient_at_point(
        key, x, t, T, delta, c, num_samples,
    ):
        n = x.shape[0]
        sigma = jnp.sqrt(jnp.maximum(delta * (T - t), 1e-30))
        t_eff = jnp.maximum(T - t, 1e-30)
        z = jax.random.normal(key, shape=(num_samples, n))
        d = x[None, :] + sigma * z
        g_vals = vmap(terminal_cost_fn)(d)
        log_w = -c * g_vals
        log_w_shifted = log_w - jnp.max(log_w)
        weights = jnp.exp(log_w_shifted)
        weights = weights / jnp.sum(weights)
        weighted_mean = jnp.sum(weights[:, None] * d, axis=0)
        return (1.0 / (t_eff * delta * c)) * (x - weighted_mean)

    return mc_gradient_at_point


# Build the default MC functions for the standard terminal cost
_mc_value_default = _make_mc_value_fn(dubins_terminal_cost)
_mc_gradient_default = _make_mc_gradient_fn(dubins_terminal_cost)


def mc_value_batch(key, eval_points, t, T, delta, c, num_samples):
    """Batch MC value evaluation."""
    M = eval_points.shape[0]
    keys = jax.random.split(key, M)
    c_arr = jnp.broadcast_to(jnp.asarray(c, dtype=jnp.float32), (M,))

    def _solve_one(k, xi, ci):
        return _mc_value_default(k, xi, t, T, delta, ci, num_samples)

    return vmap(_solve_one)(keys, eval_points, c_arr)


def mc_gradient_batch(key, eval_points, t, T, delta, c, num_samples):
    """Batch MC gradient evaluation."""
    M = eval_points.shape[0]
    keys = jax.random.split(key, M)
    c_arr = jnp.broadcast_to(jnp.asarray(c, dtype=jnp.float32), (M,))

    def _grad_one(k, xi, ci):
        return _mc_gradient_default(k, xi, t, T, delta, ci, num_samples)

    return vmap(_grad_one)(keys, eval_points, c_arr)


# ============================================================================
#  Frozen coefficient and residual
# ============================================================================

@jit
def compute_frozen_coefficient(H_vals, grad_v_sq, delta, eps=1e-8):
    return 2.0 * H_vals / (delta * jnp.maximum(grad_v_sq, eps))


@jit
def relative_residual(v_new, v_old):
    return jnp.linalg.norm(v_new - v_old) / jnp.maximum(
        jnp.linalg.norm(v_old), 1e-12
    )


# ============================================================================
#  Quasi-Linearization (Algorithm 1) -- JAX
# ============================================================================

def solve_quasi_linear_jax(
    eval_points: jnp.ndarray,
    t: float,
    T: float,
    delta: float,
    c_init: jnp.ndarray,
    hamiltonian_fn,
    num_samples: int,
    max_iters: int,
    tol: float,
    key: jax.Array,
) -> Tuple[jnp.ndarray, List[float]]:
    """Quasi-linearization iteration (Algorithm 1) using JAX."""
    v_current = vmap(dubins_terminal_cost)(eval_points)
    c_frozen = jnp.abs(c_init) + 1e-8

    history: List[float] = []

    for k in range(max_iters):
        key, subkey = jax.random.split(key)
        v_new = mc_value_batch(
            subkey, eval_points, t, T, delta, c_frozen, num_samples,
        )

        resid = float(relative_residual(v_new, v_current))
        history.append(resid)
        v_current = v_new

        if resid < tol:
            break

        key, subkey = jax.random.split(key)
        Dv = mc_gradient_batch(
            subkey, eval_points, t, T, delta, c_frozen, num_samples,
        )

        H_vals = vmap(
            lambda x, p: hamiltonian_fn(t, x, p)
        )(eval_points, Dv)

        grad_v_sq = jnp.sum(Dv ** 2, axis=-1)
        c_frozen = compute_frozen_coefficient(H_vals, grad_v_sq, delta)
        c_frozen = jnp.abs(c_frozen) + 1e-8

    return v_current, history


# ============================================================================
#  Solver class
# ============================================================================

class DubinsSolverJAX:
    """JAX-accelerated Dubins car backward reachability solver.

    Supports 'simple' and 'pursuit_evasion' modes.
    """

    def __init__(
        self,
        config: Optional[SolverConfig] = None,
        mode: str = "pursuit_evasion",
        speed: float = 1.0,
        omega_max: float = 1.0,
        target_radius: float = 0.5,
    ):
        self.cfg = config or SolverConfig(
            delta=0.08,
            num_samples=10_000,
            max_iters=15,
            tol=1e-5,
            t_start=0.0,
            t_end=1.0,
            seed=42,
        )
        self.mode = mode
        self.target_radius = target_radius

        if mode == "simple":
            self.H_fn = partial(
                dubins_simple_hamiltonian,
                speed=speed, omega_max=omega_max,
            )
        elif mode == "pursuit_evasion":
            self.H_fn = partial(
                dubins_relative_hamiltonian,
                v_p=-speed, v_e=speed, w=omega_max,
            )
        else:
            raise ValueError(f"Unknown mode: {mode}")

        self.key = jax.random.PRNGKey(self.cfg.seed)

    def _compute_initial_c(self, eval_points, t):
        """Compute c^{(0)} via autodiff of terminal cost."""
        Dg_fn = jax.grad(dubins_terminal_cost)
        Dg = vmap(Dg_fn)(eval_points)
        H_vals = vmap(lambda x, p: self.H_fn(t, x, p))(eval_points, Dg)
        Dg_sq = jnp.sum(Dg ** 2, axis=-1)
        return compute_frozen_coefficient(H_vals, Dg_sq, self.cfg.delta)

    def solve_slice(
        self,
        theta_val: float,
        grid_res: int = 40,
        domain: Tuple[float, float] = (-4.0, 4.0),
        t_eval: float = 0.0,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, List[float]]:
        """Solve on a 2D (x, y) slice at fixed theta."""
        xs = jnp.linspace(domain[0], domain[1], grid_res)
        X, Y = jnp.meshgrid(xs, xs, indexing="ij")

        theta_col = jnp.full((grid_res * grid_res, 1), theta_val)
        eval_points = jnp.concatenate(
            [X.ravel()[:, None], Y.ravel()[:, None], theta_col],
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
            hamiltonian_fn=self.H_fn,
            num_samples=self.cfg.num_samples,
            max_iters=self.cfg.max_iters,
            tol=self.cfg.tol,
            key=subkey,
        )

        V = v.reshape(grid_res, grid_res)
        return X, Y, V, history

    def solve_volume(
        self,
        grid_res: int = 25,
        spatial_domain: Tuple[float, float] = (-4.0, 4.0),
        theta_range: Tuple[float, float] = (-pi, pi),
        t_eval: float = 0.0,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, List[float]]:
        """Solve on the full 3D grid."""
        xs = jnp.linspace(spatial_domain[0], spatial_domain[1], grid_res)
        ts = jnp.linspace(theta_range[0], theta_range[1], grid_res)
        XX, YY, TT = jnp.meshgrid(xs, xs, ts, indexing="ij")

        eval_points = jnp.stack(
            [XX.ravel(), YY.ravel(), TT.ravel()], axis=-1,
        )

        c_init = self._compute_initial_c(eval_points, t_eval)

        self.key, subkey = jax.random.split(self.key)
        v, history = solve_quasi_linear_jax(
            eval_points=eval_points,
            t=t_eval,
            T=self.cfg.t_end,
            delta=self.cfg.delta,
            c_init=c_init,
            hamiltonian_fn=self.H_fn,
            num_samples=self.cfg.num_samples,
            max_iters=self.cfg.max_iters,
            tol=self.cfg.tol,
            key=subkey,
        )

        V = v.reshape(grid_res, grid_res, grid_res)
        return XX, YY, TT, V, history


# ============================================================================
#  Plotting
# ============================================================================

def plot_2d_slices(
    solver: DubinsSolverJAX,
    theta_slices: Optional[List[float]] = None,
    grid_res: int = 40,
    domain: Tuple[float, float] = (-4.0, 4.0),
    save_path: Optional[str] = None,
) -> None:
    """Plot 2D (x, y) slices of the Dubins BRT."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if theta_slices is None:
        theta_slices = [-pi / 2, 0.0, pi / 2]

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
        X, Y, V, history = solver.solve_slice(
            theta_val, grid_res=grid_res, domain=domain,
        )
        elapsed = time_mod.time() - t0
        print(f"    {len(history)} iters, {elapsed:.1f}s")

        X_np, Y_np, V_np = np.array(X), np.array(Y), np.array(V)
        ax.contourf(X_np, Y_np, V_np, levels=20, cmap="RdBu_r")
        ax.contour(X_np, Y_np, V_np, levels=[0.0], colors="k", linewidths=2)
        ax.set_title(
            rf"$\theta = {theta_val:.2f}$  ({len(history)} iters)",
            fontsize=11, fontweight="bold",
        )
        ax.set_xlabel(r"$x_1$ (m)", fontsize=11, fontweight="bold")
        ax.set_ylabel(r"$x_2$ (m)", fontsize=11, fontweight="bold")
        ax.set_aspect("equal")

    fig.suptitle(
        f"Dubins Car BRT: Quasi-Linearized Cole-Hopf (JAX)\nMode: {solver.mode}",
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
        max_iters=15,
        tol=1e-5,
        t_start=0.0,
        t_end=1.0,
        seed=42,
    )

    solver = DubinsSolverJAX(config=cfg, mode="pursuit_evasion")

    print("=" * 60)
    print("Dubins Car BRT (JAX)")
    print("=" * 60)

    t0 = time_mod.time()
    plot_2d_slices(
        solver,
        theta_slices=[-pi / 2, 0.0, pi / 2],
        grid_res=35,
        domain=(-4.0, 4.0),
        save_path=os.path.join(results_dir, "dubins_2d_slices.png"),
    )
    elapsed = time_mod.time() - t0
    print(f"Total time (JAX): {elapsed:.1f}s")
