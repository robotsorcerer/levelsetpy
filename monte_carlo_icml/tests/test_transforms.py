"""Unit tests for Cole-Hopf transforms and frozen-coefficient utilities."""

import jax
import jax.numpy as jnp
import pytest

from src.transforms import (
    cole_hopf_forward,
    cole_hopf_inverse,
    compute_frozen_coefficient,
    quasi_linear_residual,
)


class TestColeHopfRoundtrip:
    """cole_hopf_inverse(cole_hopf_forward(v, c), c) == v."""

    @pytest.mark.parametrize("c", [0.1, 1.0, 10.0])
    def test_scalar(self, c):
        v = jnp.array(2.5)
        omega = cole_hopf_forward(v, c)
        v_rec = cole_hopf_inverse(omega, c)
        assert jnp.allclose(v_rec, v, atol=1e-12)

    def test_array(self, rng):
        v = jax.random.normal(rng, shape=(50,))
        c = 1.0
        v_rec = cole_hopf_inverse(cole_hopf_forward(v, c), c)
        assert jnp.allclose(v_rec, v, atol=1e-12)

    def test_broadcast_c(self, rng):
        k1, k2 = jax.random.split(rng)
        v = jax.random.normal(k1, shape=(20,))
        c = jnp.abs(jax.random.normal(k2, shape=(20,))) + 0.1
        v_rec = cole_hopf_inverse(cole_hopf_forward(v, c), c)
        assert jnp.allclose(v_rec, v, atol=1e-5)  # float32 precision


class TestColeHopfForward:
    def test_positive_output(self):
        """exp(-c*v) is always positive."""
        v = jnp.array([-10.0, 0.0, 10.0])
        omega = cole_hopf_forward(v, 1.0)
        assert jnp.all(omega > 0)

    def test_zero_v(self):
        """omega = 1 when v = 0."""
        omega = cole_hopf_forward(jnp.array(0.0), 5.0)
        assert jnp.allclose(omega, 1.0)


class TestColeHopfInverse:
    def test_clamp_tiny_omega(self):
        """No NaN/Inf for omega near zero."""
        omega = jnp.array(0.0)
        v = cole_hopf_inverse(omega, 1.0)
        assert jnp.isfinite(v)


class TestFrozenCoefficient:
    def test_positive(self):
        """c > 0 when H > 0 and |Dv|^2 > 0."""
        c = compute_frozen_coefficient(
            H_val=jnp.array(5.0),
            grad_v_sq=jnp.array(2.0),
            delta=0.1,
        )
        assert c > 0

    def test_safeguard_zero_grad(self):
        """No NaN when |Dv|^2 = 0 (eps kicks in)."""
        c = compute_frozen_coefficient(
            H_val=jnp.array(1.0),
            grad_v_sq=jnp.array(0.0),
            delta=0.1,
        )
        assert jnp.isfinite(c)

    def test_quadratic_h_gives_one_over_delta(self):
        """For H = (1/2)|p|^2, c = 2*(0.5*|p|^2)/(delta*|p|^2) = 1/delta."""
        p_sq = 3.7
        H_val = 0.5 * p_sq
        delta = 0.05
        c = compute_frozen_coefficient(
            H_val=jnp.array(H_val),
            grad_v_sq=jnp.array(p_sq),
            delta=delta,
        )
        assert jnp.allclose(c, 1.0 / delta, atol=1e-12)


class TestQuasiLinearResidual:
    def test_zero_at_convergence(self):
        v = jnp.array([1.0, 2.0, 3.0])
        assert jnp.allclose(quasi_linear_residual(v, v), 0.0)

    def test_positive_for_different(self, rng):
        k1, k2 = jax.random.split(rng)
        v1 = jax.random.normal(k1, (10,))
        v2 = jax.random.normal(k2, (10,))
        assert quasi_linear_residual(v1, v2) > 0
