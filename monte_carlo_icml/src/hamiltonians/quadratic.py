"""Quadratic Hamiltonian  H = (1/2) |p|^2.

This is the canonical test case for which the Cole-Hopf transformation
is exact: omega = exp(-v / delta)  satisfies the heat equation.
"""

import jax.numpy as jnp
from .base import Hamiltonian


class QuadraticHamiltonian(Hamiltonian):
    """H(t, x, p) = (1/2) |p|^2."""

    def __init__(self, dim: int = 2):
        self._dim = dim

    def __call__(self, t, x, p):
        return 0.5 * jnp.sum(p ** 2, axis=-1)

    @property
    def state_dim(self) -> int:
        return self._dim

    @property
    def is_quadratic(self) -> bool:
        return True
