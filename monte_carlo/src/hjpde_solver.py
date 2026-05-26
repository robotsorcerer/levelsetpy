"""Monte Carlo heat-kernel solver for the transformed HJ PDE.

After the Cole-Hopf transformation, omega satisfies the heat equation:
    omega_t = (delta/2) Delta omega

The fundamental solution (Green's function) gives:
    omega(t, x) = E_{y ~ N(x, delta*(T-t)*I)} [ omega(T, y) ]

For the value function recovery (Lemma D.2 / lem:value in Appendix D):
    v(t, x) = -(1/c) * log E_{z ~ N(0, I)} [ exp(-c * g(x + sigma*z)) ]

where  sigma = sqrt(delta * (T - t))  and  c  is the Cole-Hopf coefficient.
"""

from typing import Callable

import jax
import jax.numpy as jnp
from jax import vmap


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
) -> jnp.ndarray:
    """Compute Dv(t, x) via importance-weighted ratio of expectations.

    Dv = E[ Dg(y) * exp(-c g(y)) ] / E[ exp(-c g(y)) ]

    where  y = x + sigma*z, z ~ N(0, I).

    Parameters
    ----------
    (same as mc_value_at_point)

    Returns
    -------
    Gradient estimate, shape (n,).
    """
    n = x.shape[0]
    sigma = jnp.sqrt(jnp.maximum(delta * (T - t), 1e-30))

    z = jax.random.normal(key, shape=(num_samples, n))
    y = x[None, :] + sigma * z

    g_vals = vmap(terminal_cost_fn)(y)              # (J,)
    Dg_fn = jax.grad(terminal_cost_fn)
    Dg_vals = vmap(Dg_fn)(y)                        # (J, n)

    # Stable importance weights via logsumexp
    log_w = -c * g_vals                             # (J,)
    log_w_shifted = log_w - jnp.max(log_w)
    weights = jnp.exp(log_w_shifted)
    weights = weights / jnp.sum(weights)            # (J,) normalized

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
) -> jnp.ndarray:
    """Evaluate Dv(t, .) at M points via vmap.

    Returns
    -------
    (M, n) array of gradient vectors.
    """
    M = eval_points.shape[0]
    keys = jax.random.split(key, M)
    c_arr = jnp.broadcast_to(jnp.asarray(c, dtype=jnp.float32), (M,))

    def _grad_one(k, xi, ci):
        return mc_gradient_at_point(
            k, xi, t, T, delta, ci, terminal_cost_fn, num_samples
        )

    return vmap(_grad_one)(keys, eval_points, c_arr)
