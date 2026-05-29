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

import functools

import jax
import jax.numpy as jnp
from jax import vmap
from jax.experimental.shard_map import shard_map
from jax.sharding import PartitionSpec as P
import time

if TYPE_CHECKING:
    from .gpu_distribution import GPUDistributor


# ---------------------------------------------------------------------------
#  shard_map-based multi-GPU kernels
#  (replaces pmap; shard_map auto-reshards inputs so there are no JIT sharding
#   cache mismatches across quasi-linear iterations)
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=16)
def _make_value_shard_fn(terminal_cost_fn, num_samples, mesh):
    """Return a jit+shard_map value kernel for the given (fn, n_samples, mesh)."""
    def _body(keys, x, c, t, T, delta):
        @functools.partial(
            shard_map,
            mesh=mesh,
            in_specs=(P("d"), P("d"), P("d"), P(), P(), P()),
            out_specs=P("d"),
            check_rep=False,
        )
        def _shard(k_loc, x_loc, c_loc, t_, T_, d_):
            def _one(k, xi, ci):
                return mc_value_at_point(k, xi, t_, T_, d_, ci, terminal_cost_fn, num_samples)
            return vmap(_one)(k_loc, x_loc, c_loc)
        return _shard(keys, x, c, t, T, delta)
    return jax.jit(_body)


@functools.lru_cache(maxsize=16)
def _make_grad_shard_fn(terminal_cost_fn, num_samples, gradient_mode, mesh):
    """Return a jit+shard_map gradient kernel for the given (fn, n_samples, mode, mesh)."""
    def _body(keys, x, c, t, T, delta):
        @functools.partial(
            shard_map,
            mesh=mesh,
            in_specs=(P("d"), P("d"), P("d"), P(), P(), P()),
            out_specs=P("d"),
            check_rep=False,
        )
        def _shard(k_loc, x_loc, c_loc, t_, T_, d_):
            def _one(k, xi, ci):
                return mc_gradient_at_point(k, xi, t_, T_, d_, ci, terminal_cost_fn, num_samples, gradient_mode)
            return vmap(_one)(k_loc, x_loc, c_loc)
        return _shard(keys, x, c, t, T, delta)
    return jax.jit(_body)


def _ensure_device_ready(devices):
    """Ensure all devices are ready for computation (blocks for synchronization)."""
    for device in devices:
        try:
            _ = jax.device_get(jnp.array(0.0, device=device))
        except Exception:
            pass  # Device may not support device_get; continue


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
    chunk_size: int = 10000,
) -> jnp.ndarray:
    """Evaluate v(t, .) at M points via pmap across available GPUs.

    Processes in chunks to manage memory; each chunk is distributed via pmap.
    Falls back to single-GPU vmap for single-device setups.

    Parameters
    ----------
    distributor : GPUDistributor
        Device distributor with auto-detected GPU count.
    chunk_size : int
        Points per chunk for streaming (default 10k).

    Returns
    -------
    (M,) array of v values.
    """
    M = eval_points.shape[0]

    if not distributor.is_multi_gpu:
        # Fallback: single GPU (uses vmap in loop for memory efficiency)
        return _mc_value_batch_chunked_vmap(
            key, eval_points, t, T, delta, c, terminal_cost_fn, num_samples, chunk_size
        )

    # Multi-GPU via shard_map: one JIT call dispatches to all devices with no
    # per-chunk Python loop over devices and no manual gather.
    c_arr = jnp.asarray(c, dtype=jnp.float32)
    if c_arr.ndim == 0:
        c_arr = jnp.broadcast_to(c_arr, (M,))
    else:
        c_arr = jnp.broadcast_to(c_arr, (M,))

    t_s = jnp.float32(t)
    T_s = jnp.float32(T)
    d_s = jnp.float32(delta)

    value_fn = _make_value_shard_fn(terminal_cost_fn, num_samples, distributor.mesh)

    # Extract seed to Python once. jax.device_get materialises ANY sharding to host,
    # so this is safe even when `key` carries P('d',) from a previous shard_map call.
    # We must not call fold_in on a distributed (2,) key: JAX applies it per-shard,
    # corrupting the PRNGKey. Derive per-chunk keys from a fresh PRNGKey instead.
    key_seed = int(jax.device_get(key).ravel()[0])

    results = []
    for chunk_idx in range(0, M, chunk_size):
        chunk_slice = slice(chunk_idx, min(chunk_idx + chunk_size, M))
        chunk_points = eval_points[chunk_slice]
        chunk_c = c_arr[chunk_slice]

        # Pad to multiple of n_devices and reshape to flat (n_devices*m_per_dev, ...).
        chunk_sharded, (n_chunk, m_per_dev) = distributor.shard_batch(chunk_points)
        c_sharded, _ = distributor.shard_batch(chunk_c)
        _ck = jax.device_get(jax.random.fold_in(jax.random.PRNGKey(key_seed), chunk_idx))
        keys_sharded = distributor.create_keys_per_device(int(_ck[0]), m_per_dev)

        chunk_flat = chunk_sharded.reshape(-1, chunk_points.shape[-1])
        c_flat = c_sharded.reshape(-1)
        keys_flat = keys_sharded.reshape(-1, 2)

        # Pre-distribute inputs across devices so shard_map reads local data.
        chunk_flat = jax.device_put(chunk_flat, distributor.row_sharding)
        c_flat = jax.device_put(c_flat, distributor.row_sharding)
        keys_flat = jax.device_put(keys_flat, distributor.row_sharding)

        # Single jit+shard_map call; output already assembled as (n_devices*m_per_dev,).
        v_flat = value_fn(keys_flat, chunk_flat, c_flat, t_s, T_s, d_s)
        results.append(v_flat[:n_chunk])

    return jnp.concatenate(results, axis=0)


def _mc_value_batch_chunked_vmap(
    key: jax.Array,
    eval_points: jnp.ndarray,
    t: float,
    T: float,
    delta: float,
    c: jnp.ndarray,
    terminal_cost_fn: Callable,
    num_samples: int,
    chunk_size: int,
) -> jnp.ndarray:
    """Single-GPU chunked vmap for value batch (used as fallback)."""
    M = eval_points.shape[0]
    c_arr = jnp.asarray(c, dtype=jnp.float32)
    if c_arr.ndim == 0:
        c_arr = jnp.broadcast_to(c_arr, (M,))
    else:
        c_arr = jnp.broadcast_to(c_arr, (M,))

    results = []
    for chunk_idx in range(0, M, chunk_size):
        chunk_slice = slice(chunk_idx, min(chunk_idx + chunk_size, M))
        chunk_points = eval_points[chunk_slice]
        chunk_c = c_arr[chunk_slice]

        key = jax.random.fold_in(key, chunk_idx)
        v_chunk = mc_value_batch(
            key, chunk_points, t, T, delta, chunk_c, terminal_cost_fn, num_samples
        )
        results.append(v_chunk)

    return jnp.concatenate(results, axis=0)


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
    chunk_size: int = 10000,
) -> jnp.ndarray:
    """Evaluate Dv(t, .) at M points via pmap across available GPUs.

    Processes in chunks to manage memory; each chunk is distributed via pmap.
    Falls back to single-GPU vmap for single-device setups.

    Parameters
    ----------
    distributor : GPUDistributor, optional
        Device distributor. If None, creates one automatically.
    chunk_size : int
        Points per chunk for streaming (default 10k).

    Returns
    -------
    (M, n) array of gradient vectors.
    """
    if distributor is None:
        from .gpu_distribution import GPUDistributor
        distributor = GPUDistributor(auto_detect=True)

    M = eval_points.shape[0]

    if not distributor.is_multi_gpu:
        # Fallback: single GPU (uses vmap in loop for memory efficiency)
        return _mc_gradient_batch_chunked_vmap(
            key, eval_points, t, T, delta, c, terminal_cost_fn, num_samples,
            gradient_mode, chunk_size
        )

    # Multi-GPU via shard_map (see mc_value_batch_distributed for rationale).
    c_arr = jnp.asarray(c, dtype=jnp.float32)
    if c_arr.ndim == 0:
        c_arr = jnp.broadcast_to(c_arr, (M,))
    else:
        c_arr = jnp.broadcast_to(c_arr, (M,))

    t_s = jnp.float32(t)
    T_s = jnp.float32(T)
    d_s = jnp.float32(delta)

    grad_fn = _make_grad_shard_fn(terminal_cost_fn, num_samples, gradient_mode, distributor.mesh)

    # See mc_value_batch_distributed for the rationale behind device_get + fresh PRNGKey.
    key_seed = int(jax.device_get(key).ravel()[0])

    results = []
    for chunk_idx in range(0, M, chunk_size):
        chunk_slice = slice(chunk_idx, min(chunk_idx + chunk_size, M))
        chunk_points = eval_points[chunk_slice]
        chunk_c = c_arr[chunk_slice]

        chunk_sharded, (n_chunk, m_per_dev) = distributor.shard_batch(chunk_points)
        c_sharded, _ = distributor.shard_batch(chunk_c)
        _ck = jax.device_get(jax.random.fold_in(jax.random.PRNGKey(key_seed), chunk_idx))
        keys_sharded = distributor.create_keys_per_device(int(_ck[0]), m_per_dev)

        chunk_flat = chunk_sharded.reshape(-1, chunk_points.shape[-1])
        c_flat = c_sharded.reshape(-1)
        keys_flat = keys_sharded.reshape(-1, 2)

        # Pre-distribute inputs across devices so shard_map reads local data.
        chunk_flat = jax.device_put(chunk_flat, distributor.row_sharding)
        c_flat = jax.device_put(c_flat, distributor.row_sharding)
        keys_flat = jax.device_put(keys_flat, distributor.row_sharding)

        # Single jit+shard_map call; output already assembled as (n_devices*m_per_dev, n).
        Dv_flat = grad_fn(keys_flat, chunk_flat, c_flat, t_s, T_s, d_s)
        results.append(Dv_flat[:n_chunk])

    return jnp.concatenate(results, axis=0)


def _mc_gradient_batch_chunked_vmap(
    key: jax.Array,
    eval_points: jnp.ndarray,
    t: float,
    T: float,
    delta: float,
    c: jnp.ndarray,
    terminal_cost_fn: Callable,
    num_samples: int,
    gradient_mode: str,
    chunk_size: int,
) -> jnp.ndarray:
    """Single-GPU chunked vmap for gradient batch (used as fallback)."""
    M = eval_points.shape[0]
    c_arr = jnp.asarray(c, dtype=jnp.float32)
    if c_arr.ndim == 0:
        c_arr = jnp.broadcast_to(c_arr, (M,))
    else:
        c_arr = jnp.broadcast_to(c_arr, (M,))

    results = []
    for chunk_idx in range(0, M, chunk_size):
        chunk_slice = slice(chunk_idx, min(chunk_idx + chunk_size, M))
        chunk_points = eval_points[chunk_slice]
        chunk_c = c_arr[chunk_slice]

        key = jax.random.fold_in(key, chunk_idx)
        Dv_chunk = mc_gradient_batch(
            key, chunk_points, t, T, delta, chunk_c, terminal_cost_fn, num_samples, gradient_mode
        )
        results.append(Dv_chunk)

    return jnp.concatenate(results, axis=0)
