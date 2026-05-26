"""HJ Reachability solver via sampling (Cole-Hopf + MC Gaussian).

Two paths:
  1. **Exact Cole-Hopf** for quadratic H = (1/2)|p|^2:
     single-pass, c = 1/delta, heat-kernel expectation.
  2. **Quasi-linearization** for general H:
     iteratively freeze c = 2H/(delta|Dv|^2) and solve.
"""

from typing import Callable, Dict, List, Tuple, Optional, TYPE_CHECKING

import jax
import jax.numpy as jnp
from jax import vmap

from .config import SolverConfig
from .hamiltonians.base import Hamiltonian
from .transforms import compute_frozen_coefficient, quasi_linear_residual
from .heat_solver import (
    mc_value_batch,
    mc_gradient_batch,
    mc_value_batch_distributed,
    mc_gradient_batch_distributed,
)

if TYPE_CHECKING:
    from .gpu_distribution import GPUDistributor


class HJReachabilitySampler:
    """Storage-free HJ reachability solver."""

    def __init__(
        self,
        hamiltonian: Hamiltonian,
        terminal_cost_fn: Callable,
        config: SolverConfig,
        distributor: Optional["GPUDistributor"] = None,
    ):
        self.H = hamiltonian
        self.g = terminal_cost_fn
        self.cfg = config
        self.distributor = distributor
        self.key = jax.random.PRNGKey(config.seed)

    # ------------------------------------------------------------------
    # Distributed/chunked evaluation helpers
    # ------------------------------------------------------------------

    def _value_batch_distributed(
        self,
        key: jax.Array,
        eval_points: jnp.ndarray,
        t: float,
        c,
    ) -> jnp.ndarray:
        """Evaluate value function with chunking and multi-GPU dispatch.

        Chunks eval_points to reduce peak memory, dispatches to pmap when
        multi-GPU available, falls back to vmap for single-GPU.
        """
        M = eval_points.shape[0]
        chunk_size = self.cfg.chunk_size

        c_arr = jnp.asarray(c)

        if M <= chunk_size:
            # Small batch: process as-is
            if self.distributor is not None and self.distributor.is_multi_gpu:
                return mc_value_batch_distributed(
                    key, eval_points, t, self.cfg.t_end,
                    self.cfg.delta, c_arr, self.g, self.cfg.num_samples,
                    self.distributor,
                )
            else:
                return mc_value_batch(
                    key, eval_points, t, self.cfg.t_end,
                    self.cfg.delta, c_arr, self.g, self.cfg.num_samples,
                )

        # Large batch: chunk and accumulate
        results = []
        for i in range(0, M, chunk_size):
            chunk = eval_points[i:i + chunk_size]
            c_chunk = c_arr[i:i + chunk_size] if c_arr.ndim > 0 else c_arr
            self.key, subkey = jax.random.split(self.key)

            if self.distributor is not None and self.distributor.is_multi_gpu:
                v_chunk = mc_value_batch_distributed(
                    subkey, chunk, t, self.cfg.t_end,
                    self.cfg.delta, c_chunk, self.g, self.cfg.num_samples,
                    self.distributor,
                )
            else:
                v_chunk = mc_value_batch(
                    subkey, chunk, t, self.cfg.t_end,
                    self.cfg.delta, c_chunk, self.g, self.cfg.num_samples,
                )
            results.append(v_chunk)

        return jnp.concatenate(results, axis=0)

    def _gradient_batch_distributed(
        self,
        key: jax.Array,
        eval_points: jnp.ndarray,
        t: float,
        c,
    ) -> jnp.ndarray:
        """Evaluate gradient with chunking and multi-GPU dispatch."""
        M = eval_points.shape[0]
        chunk_size = self.cfg.chunk_size

        c_arr = jnp.asarray(c)

        if M <= chunk_size:
            # Small batch: process as-is
            if self.distributor is not None and self.distributor.is_multi_gpu:
                return mc_gradient_batch_distributed(
                    key, eval_points, t, self.cfg.t_end,
                    self.cfg.delta, c_arr, self.g, self.cfg.num_samples,
                    self.cfg.gradient_mode, self.distributor,
                )
            else:
                return mc_gradient_batch(
                    key, eval_points, t, self.cfg.t_end,
                    self.cfg.delta, c_arr, self.g, self.cfg.num_samples,
                    self.cfg.gradient_mode,
                )

        # Large batch: chunk and accumulate
        results = []
        for i in range(0, M, chunk_size):
            chunk = eval_points[i:i + chunk_size]
            c_chunk = c_arr[i:i + chunk_size] if c_arr.ndim > 0 else c_arr
            self.key, subkey = jax.random.split(self.key)

            if self.distributor is not None and self.distributor.is_multi_gpu:
                Dv_chunk = mc_gradient_batch_distributed(
                    subkey, chunk, t, self.cfg.t_end,
                    self.cfg.delta, c_chunk, self.g, self.cfg.num_samples,
                    self.cfg.gradient_mode, self.distributor,
                )
            else:
                Dv_chunk = mc_gradient_batch(
                    subkey, chunk, t, self.cfg.t_end,
                    self.cfg.delta, c_chunk, self.g, self.cfg.num_samples,
                    self.cfg.gradient_mode,
                )
            results.append(Dv_chunk)

        return jnp.concatenate(results, axis=0)

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
        """Iterative quasi-linearization for general H (Algorithm 1 in hjgauss.pdf).

        Each Picard iteration k:
          1. Recover gradient Dv^(k-1) using c^(k-1) from the previous step.
          2. Compute H at (x, Dv^(k-1)) to form c^(k) = (2/delta) * H / |Dv|^2.
          3. Clip c^(k) to [c_min, c_max] per Assumption 2.8 (no abs — sign preserved
             to maintain convexity of the frozen linear PDE).
          4. Solve the heat equation with c^(k) frozen for this iteration.

        Returns
        -------
        v_values : (M,)
        history  : list of relative residuals per iteration.
        """
        M, n = eval_points.shape

        # Coefficient bounds per Assumption 2.8
        c_min = 1e-4
        c_max = 1e4

        # Initialize: c^(0) = 1/delta (exact Cole-Hopf baseline)
        c_current = jnp.full((M,), 1.0 / self.cfg.delta, dtype=jnp.float32)

        # Initialize value from terminal cost
        v_current = vmap(self.g)(eval_points)
        history: List[float] = []

        for _ in range(self.cfg.max_quasi_iters):
            # 1. Recover spatial gradient Dv^(k-1) using c^(k-1)
            #    (Algorithm 1 line 3: p_hat <- nabla v^(k-1) / |nabla v^(k-1)|)
            self.key, k1 = jax.random.split(self.key)
            Dv = self._gradient_batch_distributed(
                k1, eval_points, t, c_current
            )

            # 2. Evaluate Hamiltonian H^delta at current (x, Dv)
            #    (Algorithm 1 line 4: c^(k) <- (2/delta) H^delta(x, p_hat) / |Dv|^2)
            H_vals = vmap(
                lambda x, p: self.H(t, x, p)
            )(eval_points, Dv)

            # 3. Freeze coefficient c^(k) = (2/delta) * H / |Dv|^2
            #    Clip to [c_min, c_max] per Assumption 2.8.
            #    Note: clip, NOT abs — sign matters for contraction.
            grad_v_sq = jnp.sum(Dv ** 2, axis=-1)
            c_frozen = compute_frozen_coefficient(
                H_vals, grad_v_sq, self.cfg.delta
            )
            c_frozen = jnp.clip(c_frozen, c_min, c_max)

            # 4. Solve heat equation with per-point frozen c^(k)
            self.key, k2 = jax.random.split(self.key)
            v_new = self._value_batch_distributed(
                k2, eval_points, t, c_frozen
            )

            # 5. Check convergence; update c for next gradient recovery step
            residual = float(quasi_linear_residual(v_new, v_current))
            history.append(residual)
            v_current = v_new
            c_current = c_frozen  # carry frozen c forward for next gradient recovery

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

    def solve_distributed(
        self,
        eval_points: jnp.ndarray,
        t: float = 0.0,
        distributor: Optional["GPUDistributor"] = None,
    ) -> Tuple[jnp.ndarray, List[float]]:
        """Solve quasi-linear HJ equation via multi-GPU distribution.

        Automatically detects available GPUs and distributes computation.
        Falls back to single-GPU vmap if only one device available.

        Parameters
        ----------
        eval_points : jnp.ndarray, shape (M, n)
            Evaluation points.
        t : float
            Current time.
        distributor : GPUDistributor, optional
            GPU distributor. If None, creates one automatically.

        Returns
        -------
        v : jnp.ndarray, shape (M,)
            BRT value estimates.
        history : list of float
            Quasi-linearization residuals per iteration.
        """
        from .gpu_distribution import GPUDistributor

        if distributor is None:
            distributor = GPUDistributor(auto_detect=True)

        print(f"[HJReachabilitySampler] Distributed solve on {distributor}")

        # Coefficient bounds per Assumption 2.8
        c_min = 1e-4
        c_max = 1e4

        # Initialize: c^(0) = 1/delta
        M = eval_points.shape[0]
        c_current = jnp.full((M,), 1.0 / self.cfg.delta, dtype=jnp.float32)

        v_current = vmap(self.g)(eval_points)
        history = []

        for _ in range(self.cfg.max_quasi_iters):
            # 1. Recover spatial gradient using c^(k-1) (Algorithm 1 line 3)
            self.key, k1 = jax.random.split(self.key)
            Dv = mc_gradient_batch_distributed(
                k1,
                eval_points,
                t,
                self.cfg.t_end,
                self.cfg.delta,
                c_current,
                self.g,
                self.cfg.num_samples,
                self.cfg.gradient_mode,
                distributor,
            )

            # 2. Evaluate Hamiltonian at (x, Dv)
            H_vals = vmap(lambda x, p: self.H(t, x, p))(eval_points, Dv)

            # 3. Freeze coefficient c^(k) = (2/delta) * H / |Dv|^2
            #    Clip to [c_min, c_max] per Assumption 2.8 — no abs.
            grad_v_sq = jnp.sum(Dv**2, axis=-1)
            c_frozen = compute_frozen_coefficient(H_vals, grad_v_sq, self.cfg.delta)
            c_frozen = jnp.clip(c_frozen, c_min, c_max)

            # 4. Solve heat equation (distributed) with frozen c^(k)
            self.key, k2 = jax.random.split(self.key)
            v_new = mc_value_batch_distributed(
                k2,
                eval_points,
                t,
                self.cfg.t_end,
                self.cfg.delta,
                c_frozen,
                self.g,
                self.cfg.num_samples,
                distributor,
            )

            # 5. Check convergence; carry c forward for next gradient recovery
            residual = float(quasi_linear_residual(v_new, v_current))
            history.append(residual)
            v_current = v_new
            c_current = c_frozen

            if residual < self.cfg.quasi_tol:
                break

        return v_current, history
