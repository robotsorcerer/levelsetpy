"""HJ Reachability solver via sampling (Cole-Hopf + MC Gaussian).

Two paths:
  1. **Exact Cole-Hopf** for quadratic H = (1/2)|p|^2:
     single-pass, c = 1/delta, heat-kernel expectation.
  2. **Quasi-linearization** for general H:
     iteratively freeze c = 2H/(delta|Dv|^2) and solve.
"""

from typing import Callable, Dict, List, Tuple

import jax
import jax.numpy as jnp
from jax import vmap

from .config import SolverConfig
from .hamiltonians.base import Hamiltonian
from .transforms import compute_frozen_coefficient, quasi_linear_residual
from .heat_solver import mc_value_batch, mc_gradient_batch


class HJReachabilitySampler:
    """Storage-free HJ reachability solver."""

    def __init__(
        self,
        hamiltonian: Hamiltonian,
        terminal_cost_fn: Callable,
        config: SolverConfig,
    ):
        self.H = hamiltonian
        self.g = terminal_cost_fn
        self.cfg = config
        self.key = jax.random.PRNGKey(config.seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def solve(
        self, eval_points: jnp.ndarray, t: float
    ) -> jnp.ndarray:
        """Compute v(t, .) at the given evaluation points.

        Dispatches to exact or quasi-linear path based on
        ``self.H.is_quadratic``.

        Parameters
        ----------
        eval_points : (M, n) query points.
        t           : scalar time in [t_start, t_end].

        Returns
        -------
        (M,) value function estimates.
        """
        if self.H.is_quadratic:
            return self.solve_exact(eval_points, t)
        v, _ = self.solve_quasi_linear(eval_points, t)
        return v

    def solve_exact(
        self, eval_points: jnp.ndarray, t: float
    ) -> jnp.ndarray:
        """Exact Cole-Hopf path for H = (1/2)|p|^2."""
        self.key, subkey = jax.random.split(self.key)
        c = 1.0 / self.cfg.delta
        return mc_value_batch(
            subkey, eval_points, t, self.cfg.t_end,
            self.cfg.delta, c, self.g, self.cfg.num_samples,
        )

    def solve_quasi_linear(
        self,
        eval_points: jnp.ndarray,
        t: float,
    ) -> Tuple[jnp.ndarray, List[float]]:
        """Iterative quasi-linearization for general H.

        Returns
        -------
        v_values : (M,)
        history  : list of relative residuals per iteration.
        """
        M, n = eval_points.shape

        # Initialize from terminal cost
        v_current = vmap(self.g)(eval_points)
        history: List[float] = []

        for _ in range(self.cfg.max_quasi_iters):
            # 1. Recover spatial gradient at current iterate
            self.key, k1 = jax.random.split(self.key)
            # Use a moderate c for gradient recovery (start with 1/delta)
            c_grad = 1.0 / self.cfg.delta
            Dv = mc_gradient_batch(
                k1, eval_points, t, self.cfg.t_end,
                self.cfg.delta, c_grad, self.g, self.cfg.num_samples,
            )

            # 2. Evaluate Hamiltonian at current (x, Dv)
            H_vals = vmap(
                lambda x, p: self.H(t, x, p)
            )(eval_points, Dv)

            # 3. Freeze coefficient c^(k) = 2H / (delta |Dv|^2)
            grad_v_sq = jnp.sum(Dv ** 2, axis=-1)
            c_frozen = compute_frozen_coefficient(
                H_vals, grad_v_sq, self.cfg.delta
            )
            # Ensure c > 0 (flip sign if H < 0)
            c_frozen = jnp.abs(c_frozen) + 1e-8

            # 4. Solve heat equation with per-point frozen c
            self.key, k2 = jax.random.split(self.key)
            v_new = mc_value_batch(
                k2, eval_points, t, self.cfg.t_end,
                self.cfg.delta, c_frozen, self.g, self.cfg.num_samples,
            )

            # 5. Check convergence
            residual = float(quasi_linear_residual(v_new, v_current))
            history.append(residual)
            v_current = v_new

            if residual < self.cfg.quasi_tol:
                break

        return v_current, history

    def solve_backward(
        self, eval_points: jnp.ndarray
    ) -> Dict[str, jnp.ndarray]:
        """Solve backward in time from t_end to t_start.

        Returns
        -------
        dict with keys 't' (time array), 'v' (M x num_times value array).
        """
        times = jnp.linspace(
            self.cfg.t_end, self.cfg.t_start, self.cfg.time_steps
        )
        snapshots = []
        for t_i in times:
            v = self.solve(eval_points, float(t_i))
            snapshots.append(v)
        return {
            "t": times,
            "v": jnp.stack(snapshots, axis=0),
            "eval_points": eval_points,
        }
