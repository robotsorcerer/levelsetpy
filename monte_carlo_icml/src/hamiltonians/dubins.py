"""Dubins vehicle Hamiltonian for backward reachability.

Dynamics:
    x_dot     = v * cos(theta)
    y_dot     = v * sin(theta)
    theta_dot = u,    |u| <= omega_max

State: (x, y, theta) -- planar position and heading angle.

The HJI Hamiltonian (backward reachability, minimizing player controls):
    H(x, p) = v * cos(theta) * p1 + v * sin(theta) * p2
              - omega_max * |p3|

The optimal control is bang-bang:  u* = -omega_max * sign(p3).
"""

import jax.numpy as jnp
from .base import Hamiltonian


class DubinsHamiltonian(Hamiltonian):

    def __init__(
        self,
        speed: float = 1.0,
        omega_max: float = 1.0,
        smoothing_eps: float = 1e-4,
    ):
        self.speed = speed
        self.omega_max = omega_max
        self.eps = smoothing_eps

    def __call__(self, t, x, p):
        p1, p2, p3 = p[..., 0], p[..., 1], p[..., 2]
        theta = x[..., 2]
        abs_p3 = jnp.sqrt(p3 ** 2 + self.eps ** 2)
        return (self.speed * jnp.cos(theta) * p1
                + self.speed * jnp.sin(theta) * p2
                - self.omega_max * abs_p3)

    @property
    def state_dim(self) -> int:
        return 3
