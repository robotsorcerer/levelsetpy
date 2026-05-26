"""Monte Carlo heat-kernel solver for the transformed HJ PDE.

After the Cole-Hopf transformation, omega satisfies the heat equation:
    omega_t = (delta/2) Delta omega

The fundamental solution (Green's function) gives:
    omega(t, x) = E_{y ~ N(x, delta*(T-t)*I)} [ omega(T, y) ]

For the value function recovery (Lemma D.2 / lem:value in Appendix D):
    v(t, x) = -(1/c) * log E_{z ~ N(0, I)} [ exp(-c * g(x + sigma*z)) ]

where  sigma = sqrt(delta * (T - t))  and  c  is the Cole-Hopf coefficient.
"""

from typing import Callable, Optional, TYPE_CHECKING

import jax
import jax.numpy as jnp
from jax import vmap, pmap

if TYPE_CHECKING:
    from .gpu_distribution import GPUDistributor


# ---------------------------------------------------------------------------
#  Core expectation kernels
# ---------------------------------------------------------------------------

def mc_value_at_point(
    key: jax.Array,
    x: jnp.ndarray,
    t: float,
    T: float,
    delta: float,
    c: float,
    terminal_cost_fn: Callable,
    num_samples: int,
) -> jnp.ndarray:
    """Compute v(t, x) via MC Gaussian expectation with logsumexp.

    Parameters
    ----------
    key            : PRNG key.
    x              : query point, shape (n,).
    t              : current time.
    T              : terminal time.
    delta          : viscosity.
    c              : Cole-Hopf coefficient (scalar).
    terminal_cost_fn : g : R^n -> R.
    num_samples    : number of MC samples J.

    Returns
    -------
    Scalar estimate of v(t; x).
    """
    n = x.shape[0]
    sigma = jnp.sqrt(jnp.maximum(delta * (T - t), 1e-30))

    z = jax.random.normal(key, shape=(num_samples, n))
    y = x[None, :] + sigma * z                    # (J, n)
    g_vals = vmap(terminal_cost_fn)(y)             # (J,)

    # Stable logsumexp:  log( (1/J) sum exp(-c*g) )
    exponents = -c * g_vals
    max_exp = jnp.max(exponents)
    log_mean_exp = max_exp + jnp.log(
        jnp.mean(jnp.exp(exponents - max_exp))
    )
    return -(1.0 / c) * log_mean_exp


def mc_gradient_at_point(
    key: jax.Array,
    x: jnp.ndarray,
    t: float,
    T: float,
    delta: float,
    c: float,
    terminal_cost_fn: Callable,
    num_samples: int,
    gradient_mode: str = "b17",
) -> jnp.ndarray:
    """Compute Dv(t, x) via importance-weighted ratio of expectations.

    Two estimators available:

    Corollary B.5 (cor:mc_gradient / eq:mc_grad from hjgauss.pdf, default):
        Dv = (1/(t_eff*delta*c)) * (x - E[y*w] / E[w])

    Autodiff (ICML variant):
        Dv = E[ Dg(y) * exp(-c g(y)) ] / E[ exp(-c g(y)) ]

    where y = x + sigma*z, z ~ N(0, I), w = exp(-c*g(y)).

    Parameters
    ----------
    gradient_mode : str, default "b17"
        Choice of gradient estimator. "b17" uses Corollary B.5 (eq:mc_grad), "autodiff" uses autodiff.

    Returns
    -------
    Gradient estimate, shape (n,).
    """
    n = x.shape[0]
    sigma = jnp.sqrt(jnp.maximum(delta * (T - t), 1e-30))
    t_eff = jnp.maximum(T - t, 1e-30)

    z = jax.random.normal(key, shape=(num_samples, n))
    y = x[None, :] + sigma * z  # (J, n)

    g_vals = vmap(terminal_cost_fn)(y)  # (J,)

    # Stable importance weights via logsumexp
    log_w = -c * g_vals  # (J,)
    log_w_shifted = log_w - jnp.max(log_w)
    weights = jnp.exp(log_w_shifted)
    weights = weights / jnp.sum(weights)  # (J,) normalized

    if gradient_mode == "b17":
        # Corollary B.5 (eq:mc_grad): (1/(t_eff*delta*c)) * (x - weighted_mean(y))
        weighted_mean = jnp.sum(weights[:, None] * y, axis=0)
        return (1.0 / (t_eff * delta * c)) * (x - weighted_mean)
    else:  # "autodiff"
        # ICML variant: weighted importance gradient of g
        Dg_fn = jax.grad(terminal_cost_fn)
        Dg_vals = vmap(Dg_fn)(y)  # (J, n)
        return jnp.sum(weights[:, None] * Dg_vals, axis=0)


# ---------------------------------------------------------------------------
#  Batched wrappers (vmap over evaluation points)
# ---------------------------------------------------------------------------

def mc_value_batch(
    key: jax.Array,
    eval_points: jnp.ndarray,
    t: float,
    T: float,
    delta: float,
    c: jnp.ndarray,
    terminal_cost_fn: Callable,
    num_samples: int,
) -> jnp.ndarray:
    """Evaluate v(t, .) at M points via vmap.

    Parameters
    ----------
    key         : PRNG key.
    eval_points : (M, n) array of query points.
    c           : scalar or (M,) array of Cole-Hopf coefficients.

    Returns
    -------
    (M,) array of v values.
    """
    M = eval_points.shape[0]
    keys = jax.random.split(key, M)

    # If c is scalar, broadcast to (M,)
    c_arr = jnp.broadcast_to(jnp.asarray(c, dtype=jnp.float32), (M,))

    def _solve_one(k, xi, ci):
        return mc_value_at_point(
            k, xi, t, T, delta, ci, terminal_cost_fn, num_samples
        )

    return vmap(_solve_one)(keys, eval_points, c_arr)


def mc_gradient_batch(
    key: jax.Array,
    eval_points: jnp.ndarray,
    t: float,
    T: float,
    delta: float,
    c: jnp.ndarray,
    terminal_cost_fn: Callable,
    num_samples: int,
    gradient_mode: str = "b17",
) -> jnp.ndarray:
    """Evaluate Dv(t, .) at M points via vmap.

    Parameters
    ----------
    gradient_mode : str, default "b17"
        Choice of gradient estimator. "b17" uses Eq. B.17, "autodiff" uses autodiff.

    Returns
    -------
    (M, n) array of gradient vectors.
    """
    M = eval_points.shape[0]
    keys = jax.random.split(key, M)
    c_arr = jnp.broadcast_to(jnp.asarray(c, dtype=jnp.float32), (M,))

    def _grad_one(k, xi, ci):
        return mc_gradient_at_point(
            k, xi, t, T, delta, ci, terminal_cost_fn, num_samples, gradient_mode
        )

    return vmap(_grad_one)(keys, eval_points, c_arr)


# ============================================================================
#  Multi-GPU Distributed Variants (pmap)
# ============================================================================


def mc_value_batch_distributed(
    key: jax.Array,
    eval_points: jnp.ndarray,
    t: float,
    T: float,
    delta: float,
    c: jnp.ndarray,
    terminal_cost_fn: Callable,
    num_samples: int,
    distributor: "GPUDistributor",
) -> jnp.ndarray:
    """Evaluate v(t, .) at M points via pmap across available GPUs.

    Automatically falls back to single-GPU vmap if only one device available.

    Parameters
    ----------
    distributor : GPUDistributor
        Device distributor with auto-detected GPU count.

    Returns
    -------
    (M,) array of v values.
    """
    if not distributor.is_multi_gpu:
        # Fallback: single GPU
        return mc_value_batch(
            key, eval_points, t, T, delta, c, terminal_cost_fn, num_samples
        )

    # Multi-GPU: shard and pmap
    eval_points_sharded, orig_shape = distributor.shard_batch(eval_points)
    M, M_per_device = orig_shape

    # Broadcast c if scalar
    c_arr = jnp.broadcast_to(jnp.asarray(c, dtype=jnp.float32), (M,))
    c_sharded, _ = distributor.shard_batch(c_arr)

    # Create keys for each device
    keys_sharded = distributor.create_keys_per_device(int(key[0]), M_per_device)

    def _solve_one_distributed(k_per_device, x_per_device, c_per_device):
        """Vectorized solve on one device (called by pmap)."""
        def _solve_one(k, xi, ci):
            return mc_value_at_point(
                k, xi, t, T, delta, ci, terminal_cost_fn, num_samples
            )
        return vmap(_solve_one)(k_per_device, x_per_device, c_per_device)

    # pmap over devices (axis 0)
    pmapped_solve = pmap(_solve_one_distributed, axis_name="i")
    v_sharded = pmapped_solve(keys_sharded, eval_points_sharded, c_sharded)

    # Unshard and remove padding
    v = distributor.unshard_batch(v_sharded, orig_shape)
    return v


def mc_gradient_batch_distributed(
    key: jax.Array,
    eval_points: jnp.ndarray,
    t: float,
    T: float,
    delta: float,
    c: jnp.ndarray,
    terminal_cost_fn: Callable,
    num_samples: int,
    gradient_mode: str = "b17",
    distributor: Optional["GPUDistributor"] = None,
) -> jnp.ndarray:
    """Evaluate Dv(t, .) at M points via pmap across available GPUs.

    Automatically falls back to single-GPU vmap if only one device available.

    Parameters
    ----------
    distributor : GPUDistributor, optional
        Device distributor. If None, creates one automatically.

    Returns
    -------
    (M, n) array of gradient vectors.
    """
    if distributor is None:
        from .gpu_distribution import GPUDistributor
        distributor = GPUDistributor(auto_detect=True)

    if not distributor.is_multi_gpu:
        # Fallback: single GPU
        return mc_gradient_batch(
            key, eval_points, t, T, delta, c, terminal_cost_fn, num_samples, gradient_mode
        )

    # Multi-GPU: shard and pmap
    eval_points_sharded, orig_shape = distributor.shard_batch(eval_points)
    M, M_per_device = orig_shape

    # Broadcast c if scalar
    c_arr = jnp.broadcast_to(jnp.asarray(c, dtype=jnp.float32), (M,))
    c_sharded, _ = distributor.shard_batch(c_arr)

    # Create keys for each device
    keys_sharded = distributor.create_keys_per_device(int(key[0]), M_per_device)

    def _grad_one_distributed(k_per_device, x_per_device, c_per_device):
        """Vectorized gradient on one device (called by pmap)."""
        def _grad_one(k, xi, ci):
            return mc_gradient_at_point(
                k, xi, t, T, delta, ci, terminal_cost_fn, num_samples, gradient_mode
            )
        return vmap(_grad_one)(k_per_device, x_per_device, c_per_device)

    # pmap over devices (axis 0)
    pmapped_grad = pmap(_grad_one_distributed, axis_name="i")
    Dv_sharded = pmapped_grad(keys_sharded, eval_points_sharded, c_sharded)

    # Unshard and remove padding
    Dv = distributor.unshard_batch(Dv_sharded, orig_shape)
    return Dv
