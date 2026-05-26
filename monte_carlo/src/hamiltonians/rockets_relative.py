"""Two-rockets pursuit-evasion Hamiltonian in relative coordinates.

State: x = (x, z, theta)
  - (x, z): relative position of evader w.r.t. pursuer
  - theta = u_p - u_e: relative thrust inclination

Dynamics (Section 4.1 of the NeurIPS 2026 paper):
    x_dot     = a_p cos(theta) + u_e * x
    z_dot     = a_p sin(theta) + a_e + u_e * x - g
    theta_dot = u_p - u_e

Hamiltonian (Equations 22-23):
    H(x, p) = -a * p1 * cos(theta) - p2 * (g - a - a*sin(theta))
              - u_bar * |p1*x + p3| + u_bar * |p2*x + p3|

where a = a_p = a_e, u_bar = max(|u_p|, |u_e|).
"""

import jax.numpy as jnp
from .base import Hamiltonian


class RocketsRelativeHamiltonian(Hamiltonian):

    def __init__(
        self,
        a: float = 1.0,
        g: float = 32.0,
        u_bound: float = 1.0,
        smoothing_eps: float = 1e-4,
    ):
        self.a = a
        self.grav = g
        self.u_bound = u_bound
        self.eps = smoothing_eps

    def __call__(self, t, x, p):
        p1, p2, p3 = p[..., 0], p[..., 1], p[..., 2]
        x_pos = x[..., 0]
        theta = x[..., 2]

        smooth_abs = lambda z: jnp.sqrt(z ** 2 + self.eps ** 2)

        term1 = -self.a * p1 * jnp.cos(theta)
        term2 = -p2 * (self.grav - self.a - self.a * jnp.sin(theta))
        term3 = -self.u_bound * smooth_abs(p1 * x_pos + p3) # smooth_abs was torch.smooth in levelsetpy
        term4 = self.u_bound * smooth_abs(p2 * x_pos + p3)
        return term1 + term2 + term3 + term4

    @property
    def state_dim(self) -> int:
        return 3
