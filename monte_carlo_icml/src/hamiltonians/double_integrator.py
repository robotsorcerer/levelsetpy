"""Double-integrator Hamiltonian.

Dynamics:  x1_dot = x2,  x2_dot = u,  |u| <= u_bound.
State:     x = (x1, x2)  -- position and velocity.

The HJI PDE Hamiltonian (for backward reachability) is:

    H(x, p) = max_u min_w  <f(x, u, w), p>

For the double integrator with no disturbance:
    H(x, p) = -p1 * x2 - u_bound * |p2|

The optimal control is bang-bang:  u* = -u_bound * sign(p2).
"""

import jax.numpy as jnp
from .base import Hamiltonian


class DoubleIntegratorHamiltonian(Hamiltonian):

    def __init__(self, u_bound: float = 1.0, smoothing_eps: float = 1e-4):
        self.u_bound = u_bound
        self.eps = smoothing_eps

    def __call__(self, t, x, p):
        p1, p2 = p[..., 0], p[..., 1]
        x2 = x[..., 1]
        abs_p2 = jnp.sqrt(p2 ** 2 + self.eps ** 2)
        return -p1 * x2 - self.u_bound * abs_p2

    @property
    def state_dim(self) -> int:
        return 2
