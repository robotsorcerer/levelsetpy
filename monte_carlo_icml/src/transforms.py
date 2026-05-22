"""Cole-Hopf transformation and frozen-coefficient utilities.

The viscous HJ PDE is:
    v_t + H(t, x, Dv) = (delta/2) * Delta v        (HJ-Visc)

For H = (1/2)|p|^2, the exact Cole-Hopf is  omega = exp(-v / delta),
giving  omega_t = (delta/2) Delta omega  (heat equation).

For general H, we freeze c = 2H / (delta |Dv|^2) at the current iterate
and solve the heat equation as an approximate step, then update.
"""

import jax
import jax.numpy as jnp


@jax.jit
def cole_hopf_forward(v: jnp.ndarray, c: jnp.ndarray) -> jnp.ndarray:
    """omega = exp(-c * v).

    Parameters
    ----------
    v : value function, any shape.
    c : coefficient, broadcastable to v.
    """
    return jnp.exp(-c * v)


@jax.jit
def cole_hopf_inverse(omega: jnp.ndarray, c: jnp.ndarray) -> jnp.ndarray:
    """v = -(1/c) * log(omega), clamped for stability.

    Parameters
    ----------
    omega : transformed variable, any shape.
    c     : coefficient, broadcastable to omega.  Must be > 0.
    """
    omega_safe = jnp.maximum(omega, 1e-30)
    return -(1.0 / c) * jnp.log(omega_safe)


@jax.jit
def compute_frozen_coefficient(
    H_val: jnp.ndarray,
    grad_v_sq: jnp.ndarray,
    delta: float,
    eps: float = 1e-8,
) -> jnp.ndarray:
    """c(t, x) = 2 H / (delta |Dv|^2).

    Safeguarded against division by zero in |Dv|^2.
    """
    return 2.0 * H_val / (delta * jnp.maximum(grad_v_sq, eps))


@jax.jit
def quasi_linear_residual(
    v_new: jnp.ndarray, v_old: jnp.ndarray
) -> jnp.ndarray:
    """Relative L2 residual between successive iterates."""
    return jnp.linalg.norm(v_new - v_old) / jnp.maximum(
        jnp.linalg.norm(v_old), 1e-12
    )
