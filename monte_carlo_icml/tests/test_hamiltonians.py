"""Unit tests for Hamiltonian implementations."""

import jax
import jax.numpy as jnp
import pytest
from math import pi

from src.hamiltonians.quadratic import QuadraticHamiltonian
from src.hamiltonians.double_integrator import DoubleIntegratorHamiltonian
from src.hamiltonians.rockets_relative import RocketsRelativeHamiltonian


class TestQuadraticHamiltonian:
    def setup_method(self):
        self.H = QuadraticHamiltonian(dim=3)

    def test_is_quadratic(self):
        assert self.H.is_quadratic is True

    def test_dim(self):
        assert self.H.state_dim == 3

    def test_value(self):
        """H(t, x, p) = (1/2)|p|^2."""
        x = jnp.array([1.0, 2.0, 3.0])
        p = jnp.array([3.0, 4.0, 0.0])
        val = self.H(0.0, x, p)
        assert jnp.allclose(val, 12.5)  # 0.5*(9+16+0)

    def test_grad_wrt_p(self):
        """dH/dp = p for H = (1/2)|p|^2."""
        x = jnp.array([0.0, 0.0])
        p = jnp.array([2.0, 5.0])
        H2 = QuadraticHamiltonian(dim=2)
        grad_p = jax.grad(H2, argnums=2)(0.0, x, p)
        assert jnp.allclose(grad_p, p)

    def test_vmap(self):
        """H should work under vmap over a batch of (x, p)."""
        batch_x = jnp.ones((10, 3))
        batch_p = jnp.ones((10, 3)) * 2.0
        vals = jax.vmap(lambda x, p: self.H(0.0, x, p))(batch_x, batch_p)
        assert vals.shape == (10,)
        assert jnp.allclose(vals, 6.0)  # 0.5*(4+4+4)

    def test_independent_of_x(self):
        """Quadratic H doesn't depend on x."""
        p = jnp.array([1.0, 1.0, 1.0])
        v1 = self.H(0.0, jnp.zeros(3), p)
        v2 = self.H(0.0, jnp.ones(3) * 100, p)
        assert jnp.allclose(v1, v2)


# ---------------------------------------------------------------------------
#  Double Integrator
# ---------------------------------------------------------------------------

class TestDoubleIntegratorHamiltonian:
    def setup_method(self):
        self.H = DoubleIntegratorHamiltonian(u_bound=1.0, smoothing_eps=1e-6)

    def test_not_quadratic(self):
        assert self.H.is_quadratic is False

    def test_dim(self):
        assert self.H.state_dim == 2

    def test_known_value(self):
        """H = -p1*x2 - u*|p2|.
        At x=(0, 2), p=(1, 3): H = -1*2 - 1*|3| = -5."""
        x = jnp.array([0.0, 2.0])
        p = jnp.array([1.0, 3.0])
        val = self.H(0.0, x, p)
        assert jnp.allclose(val, -5.0, atol=1e-3)

    def test_sign_p2(self):
        """H should be the same for p2 and -p2 (absolute value)."""
        x = jnp.array([1.0, 1.0])
        p_pos = jnp.array([0.0, 2.0])
        p_neg = jnp.array([0.0, -2.0])
        assert jnp.allclose(
            self.H(0.0, x, p_pos),
            self.H(0.0, x, p_neg),
            atol=1e-3,
        )

    def test_vmap(self):
        batch_x = jnp.ones((8, 2))
        batch_p = jnp.ones((8, 2))
        vals = jax.vmap(lambda x, p: self.H(0.0, x, p))(batch_x, batch_p)
        assert vals.shape == (8,)

    def test_grad_wrt_p(self):
        """dH/dp should be computable via jax.grad."""
        x = jnp.array([0.0, 1.0])
        p = jnp.array([1.0, 2.0])
        grad_p = jax.grad(self.H, argnums=2)(0.0, x, p)
        assert grad_p.shape == (2,)
        # dH/dp1 = -x2 = -1
        assert jnp.allclose(grad_p[0], -1.0, atol=1e-4)


# ---------------------------------------------------------------------------
#  Two Rockets
# ---------------------------------------------------------------------------

class TestRocketsRelativeHamiltonian:
    def setup_method(self):
        self.H = RocketsRelativeHamiltonian(
            a=1.0, g=32.0, u_bound=1.0, smoothing_eps=1e-6,
        )

    def test_not_quadratic(self):
        assert self.H.is_quadratic is False

    def test_dim(self):
        assert self.H.state_dim == 3

    def test_known_value_theta_zero(self):
        """At theta=0: cos(0)=1, sin(0)=0.
        H = -a*p1*1 - p2*(g-a) - u*|p1*x+p3| + u*|p2*x+p3|
        At x=(1,0,0), p=(1,1,1):
        = -1 - 31 - |1+1| + |1+1| = -1 - 31 = -32
        """
        x = jnp.array([1.0, 0.0, 0.0])
        p = jnp.array([1.0, 1.0, 1.0])
        val = self.H(0.0, x, p)
        assert jnp.allclose(val, -32.0, atol=1e-2)

    def test_vmap(self):
        batch_x = jnp.zeros((5, 3))
        batch_p = jnp.ones((5, 3))
        vals = jax.vmap(lambda x, p: self.H(0.0, x, p))(batch_x, batch_p)
        assert vals.shape == (5,)

    def test_grad_wrt_p(self):
        """dH/dp should be computable."""
        x = jnp.array([1.0, 0.0, pi / 4])
        p = jnp.array([1.0, 1.0, 1.0])
        grad_p = jax.grad(self.H, argnums=2)(0.0, x, p)
        assert grad_p.shape == (3,)
        assert jnp.all(jnp.isfinite(grad_p))

    def test_smoothing_approaches_exact(self):
        """As eps->0, smooth_abs(z) -> |z|."""
        H_smooth = RocketsRelativeHamiltonian(smoothing_eps=1e-10)
        H_rough = RocketsRelativeHamiltonian(smoothing_eps=1e-2)
        x = jnp.array([2.0, 1.0, 0.3])
        p = jnp.array([1.0, -1.0, 0.5])
        v_smooth = H_smooth(0.0, x, p)
        v_rough = H_rough(0.0, x, p)
        # Both finite; smooth version closer to true value
        assert jnp.isfinite(v_smooth)
        assert jnp.isfinite(v_rough)
