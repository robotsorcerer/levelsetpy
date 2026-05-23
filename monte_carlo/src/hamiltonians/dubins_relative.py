"""Dubins vehicle pursuit-evasion Hamiltonian in relative coordinates.

Two Dubins vehicles (pursuer and evader) in the Merz (1972) formulation.

State: x = (x1, x2, x3)
  - (x1, x2): relative position of evader w.r.t. pursuer
  - x3: relative heading angle

Dynamics:
    x1_dot = -v_e + v_p cos(x3) + w_e x2
    x2_dot = -v_p sin(x3) - w_e x1
    x3_dot = -w_p - w_e

Hamiltonian (Isaacs, pursuer minimizes, evader maximizes):
    H(x, p) = p1 (v_e - v_p cos(x3)) - p2 (v_p sin(x3))
              - w |p1 x2 - p2 x1 - p3| + w |p3|

where v_p, v_e are linear speeds and w is the angular speed bound
(assumed equal for pursuer and evader: w_p = w_e = w).

This matches the levelsetpy DubinsVehicleRel implementation exactly.
"""

import jax.numpy as jnp
from .base import Hamiltonian


class DubinsRelativeHamiltonian(Hamiltonian):

    def __init__(
        self,
        v_p: float = -1.0,
        v_e: float = 1.0,
        w: float = 1.0,
        smoothing_eps: float = 1e-4,
    ):
        self.v_p = v_p
        self.v_e = v_e
        self.w = w
        self.eps = smoothing_eps

    def __call__(self, t, x, p):
        p1, p2, p3 = p[..., 0], p[..., 1], p[..., 2]
        x1 = x[..., 0]
        x2 = x[..., 1]
        x3 = x[..., 2]

        smooth_abs = lambda z: jnp.sqrt(z ** 2 + self.eps ** 2)

        term1 = p1 * (self.v_e - self.v_p * jnp.cos(x3))
        term2 = -p2 * (self.v_p * jnp.sin(x3))
        term3 = -self.w * smooth_abs(p1 * x2 - p2 * x1 - p3)
        term4 = self.w * smooth_abs(p3)
        return term1 + term2 + term3 + term4

    @property
    def state_dim(self) -> int:
        return 3
