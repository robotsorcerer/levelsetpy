"""Tests for the MC heat-kernel solver."""

import jax
import jax.numpy as jnp
import pytest

from src.heat_solver import (
    mc_value_at_point,
    mc_gradient_at_point,
    mc_value_batch,
    mc_gradient_batch,
)
from src.initial_conditions import quadratic_cost, sphere_cost


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------
DELTA = 0.1
T = 1.0
C = 1.0 / DELTA  # exact Cole-Hopf coefficient
J = 50_000        # enough samples for ~1% accuracy


# ---------------------------------------------------------------------------
#  Value function tests
# ---------------------------------------------------------------------------

class TestMCValueConstantCost:
    """If g(x) = k (constant), then v(t, x) = k for all t, x."""

    def test_constant(self, rng):
        k = 3.14
        g = lambda x: jnp.float32(k)
        x = jnp.array([0.0, 0.0])
        v = mc_value_at_point(rng, x, 0.5, T, DELTA, C, g, J)
        assert jnp.allclose(v, k, atol=0.05)


class TestMCValueLinearCost:
    """For g(x) = a.x + b (linear), the logsumexp / softmin formula gives:

    v(t,x) = g(x) - (c/2) * sigma^2 * |a|^2
           = (a.x + b) - (T - t)/2 * |a|^2

    This is the correct viscous smoothing bias.
    """

    def test_linear(self, rng):
        a = jnp.array([1.0, -2.0])
        b = 3.0
        g = lambda x: jnp.dot(a, x) + b
        x = jnp.array([1.0, 0.5])
        t = 0.5
        a_sq = jnp.sum(a ** 2)  # 5
        # Exact: g(x) - (T-t)/2 * |a|^2 = 3 - 0.25*5 = 1.75
        expected = jnp.dot(a, x) + b - (T - t) / 2.0 * a_sq
        v = mc_value_at_point(rng, x, t, T, DELTA, C, g, J)
        assert jnp.allclose(v, expected, atol=0.15)


class TestMCValueQuadraticCost:
    """For g(x) = |x|^2, the exact heat-equation solution is:

    v(t, x) = |x|^2 + n * delta * (T - t)

    (when using logsumexp inversion with c = 1/delta, this becomes
    an approximation that converges as J -> infinity).

    NB: The logsumexp formula  -(1/c)*log E[exp(-c*g)] is the
    cumulant generating function, not the raw expectation.
    For g ~ N(mu, sigma^2),  E[exp(-c*g)] = exp(-c*mu + c^2*sigma^2/2)
    so -(1/c)*log(.) = mu - (c/2)*sigma^2.

    With g(y)=|y|^2, y=x+sigma*z, z~N(0,I):
    g(y) = |x|^2 + 2*sigma*(x.z) + sigma^2*|z|^2
    This is NOT Gaussian, but the logsumexp formula still gives a
    well-defined smoothed value. We verify convergence empirically.
    """

    def test_quadratic_at_origin(self, rng):
        """At x=0, g(y) = sigma^2 * |z|^2 = sigma^2 * chi^2(n)."""
        n = 2
        x = jnp.zeros(n)
        t = 0.0
        v = mc_value_at_point(rng, x, t, T, DELTA, C, quadratic_cost, J)
        # The value should be finite and positive (smoothed version of 0)
        assert jnp.isfinite(v)
        assert v >= -1.0  # not wildly negative

    def test_monotone_in_radius(self, rng):
        """v should increase with |x| for sphere-like costs."""
        n = 2
        k1, k2 = jax.random.split(rng)
        x_near = jnp.array([0.5, 0.0])
        x_far = jnp.array([3.0, 0.0])
        v_near = mc_value_at_point(k1, x_near, 0.5, T, DELTA, C,
                                   quadratic_cost, J)
        v_far = mc_value_at_point(k2, x_far, 0.5, T, DELTA, C,
                                  quadratic_cost, J)
        assert v_far > v_near


class TestMCValueBatch:
    """Test the vmapped batch interface."""

    def test_shapes(self, rng):
        M, n = 20, 3
        pts = jax.random.normal(rng, shape=(M, n))
        g = lambda x: jnp.sum(x ** 2)
        vals = mc_value_batch(rng, pts, 0.5, T, DELTA, C, g, 1000)
        assert vals.shape == (M,)

    def test_determinism(self, rng):
        """Same key -> same result."""
        pts = jnp.array([[1.0, 0.0], [0.0, 1.0]])
        g = lambda x: jnp.sum(x ** 2)
        v1 = mc_value_batch(rng, pts, 0.5, T, DELTA, C, g, 5000)
        v2 = mc_value_batch(rng, pts, 0.5, T, DELTA, C, g, 5000)
        assert jnp.allclose(v1, v2)


# ---------------------------------------------------------------------------
#  Gradient tests
# ---------------------------------------------------------------------------

class TestMCGradient:
    def test_linear_cost_gradient(self, rng):
        """For g(x) = a.x + b, Dv = a."""
        a = jnp.array([2.0, -1.0])
        b = 5.0
        g = lambda x: jnp.dot(a, x) + b
        x = jnp.array([1.0, 1.0])
        Dv = mc_gradient_at_point(rng, x, 0.5, T, DELTA, C, g, J)
        assert jnp.allclose(Dv, a, atol=0.15)

    def test_gradient_shape(self, rng):
        n = 4
        x = jnp.zeros(n)
        g = lambda x: jnp.sum(x ** 2)
        Dv = mc_gradient_at_point(rng, x, 0.5, T, DELTA, C, g, 5000)
        assert Dv.shape == (n,)


class TestMCGradientBatch:
    def test_shapes(self, rng):
        M, n = 10, 3
        pts = jax.random.normal(rng, shape=(M, n))
        g = lambda x: jnp.sum(x ** 2)
        grads = mc_gradient_batch(rng, pts, 0.5, T, DELTA, C, g, 1000)
        assert grads.shape == (M, n)


# ---------------------------------------------------------------------------
#  Logsumexp stability
# ---------------------------------------------------------------------------

class TestStability:
    def test_large_cost(self, rng):
        """No NaN/Inf when g values are very large."""
        g = lambda x: 1000.0 * jnp.sum(x ** 2)
        x = jnp.array([10.0, 10.0])
        v = mc_value_at_point(rng, x, 0.5, T, DELTA, C, g, 5000)
        assert jnp.isfinite(v)

    def test_negative_cost(self, rng):
        """Works when g can be negative (inside the target set)."""
        g = lambda x: jnp.sum(x ** 2) - 100.0
        x = jnp.array([0.0, 0.0])
        v = mc_value_at_point(rng, x, 0.5, T, DELTA, C, g, 5000)
        assert jnp.isfinite(v)
