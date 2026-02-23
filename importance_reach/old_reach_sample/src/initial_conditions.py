"""Terminal cost / target-set functions (signed distance).

Convention: negative inside the target set, positive outside.
All functions operate on a single state vector x of shape (n,)
so that they can be composed with ``jax.vmap`` and ``jax.grad``.
"""

import jax.numpy as jnp
from typing import Optional


def sphere_cost(x: jnp.ndarray, center: Optional[jnp.ndarray] = None,
                radius: float = 1.0) -> jnp.ndarray:
    """Signed distance to a sphere  ||x - center|| - radius."""
    if center is None:
        center = jnp.zeros_like(x)
    return jnp.linalg.norm(x - center) - radius


def cylinder_cost(x: jnp.ndarray, axis_align: int = 2,
                  center: Optional[jnp.ndarray] = None,
                  radius: float = 1.5) -> jnp.ndarray:
    """Signed distance to an axis-aligned cylinder.

    The dimension ``axis_align`` is ignored (cylinder is infinite along
    that axis).  For the two-rockets problem, ``axis_align=2``
    (the theta dimension).
    """
    if center is None:
        center = jnp.zeros_like(x)
    diff = x - center
    # Zero out the aligned dimension
    mask = jnp.ones(x.shape[-1]).at[axis_align].set(0.0)
    return jnp.linalg.norm(diff * mask) - radius


def quadratic_cost(x: jnp.ndarray) -> jnp.ndarray:
    """g(x) = |x|^2.  Useful for exact heat-equation verification."""
    return jnp.sum(x ** 2)
