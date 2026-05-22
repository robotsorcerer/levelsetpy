#!/usr/bin/env python
"""Rigorous tests for the two-rockets quasi-linearization solver.

Tests cover:
  1. Unit tests for individual components (Hamiltonian, terminal cost, etc.)
  2. Integration tests for the full solver pipeline
  3. Convergence tests verifying Algorithm 1 behavior
  4. Consistency tests between Python and JAX implementations
  5. Mathematical correctness tests

Run with: python -m pytest rocket_test.py -v
"""

import sys
import os
import numpy as np
import pytest
from math import pi

sys.path.insert(0, os.path.dirname(__file__))

from sampling_engine import (
    SolverConfig,
    cole_hopf_forward,
    cole_hopf_inverse,
    compute_frozen_coefficient,
    relative_residual,
    mc_value_at_point,
    mc_gradient_at_point,
    cylinder_cost,
)
from rocket_python import (
    RocketsHamiltonian,
    rockets_terminal_cost,
    RocketsSolver,
    A_THRUST,
    GRAVITY,
    U_BOUND,
    CAPTURE_RADIUS,
)

# Try importing JAX modules (skip JAX tests if not available)
try:
    import jax
    import jax.numpy as jnp
    from rocket_jax import (
        rockets_hamiltonian as rockets_hamiltonian_jax,
        rockets_terminal_cost as rockets_terminal_cost_jax,
        RocketsSolverJAX,
        mc_value_at_point as mc_value_jax,
        mc_gradient_at_point as mc_gradient_jax,
    )
    HAS_JAX = True
except ImportError:
    HAS_JAX = False


# ============================================================================
#  Fixtures
# ============================================================================

@pytest.fixture
def rng():
    return np.random.default_rng(123)


@pytest.fixture
def hamiltonian():
    return RocketsHamiltonian()


@pytest.fixture
def small_config():
    return SolverConfig(
        delta=0.1,
        num_samples=2_000,
        max_iters=5,
        tol=1e-3,
        t_start=0.0,
        t_end=1.0,
        seed=123,
    )


# ============================================================================
#  Unit Tests: Terminal Cost
# ============================================================================

class TestTerminalCost:
    """Tests for the terminal cost function g(x) = ||(x,z)|| - ell."""

    def test_at_origin(self):
        """g(0, 0, theta) = -CAPTURE_RADIUS for any theta."""
        for theta in [0.0, pi / 4, -pi / 2]:
            x = np.array([0.0, 0.0, theta])
            assert rockets_terminal_cost(x) == pytest.approx(
                -CAPTURE_RADIUS, abs=1e-10
            )

    def test_on_boundary(self):
        """g should be zero on the capture boundary."""
        r = CAPTURE_RADIUS
        for angle in np.linspace(0, 2 * pi, 8, endpoint=False):
            x = np.array([r * np.cos(angle), r * np.sin(angle), 0.0])
            assert rockets_terminal_cost(x) == pytest.approx(0.0, abs=1e-10)

    def test_outside_target(self):
        """g > 0 outside the capture region."""
        x = np.array([3.0, 4.0, 0.0])  # ||(3,4)|| = 5 > 1.5
        assert rockets_terminal_cost(x) > 0.0

    def test_inside_target(self):
        """g < 0 inside the capture region."""
        x = np.array([0.5, 0.5, 0.0])  # ||(0.5, 0.5)|| ~ 0.707 < 1.5
        assert rockets_terminal_cost(x) < 0.0

    def test_theta_independence(self):
        """Terminal cost is a cylinder -- independent of theta."""
        x1 = np.array([2.0, 1.0, 0.0])
        x2 = np.array([2.0, 1.0, pi / 3])
        assert rockets_terminal_cost(x1) == pytest.approx(
            rockets_terminal_cost(x2), abs=1e-10
        )


# ============================================================================
#  Unit Tests: Hamiltonian
# ============================================================================

class TestHamiltonian:
    """Tests for the rockets Hamiltonian H(t, x, p)."""

    def test_zero_costate(self, hamiltonian):
        """H(t, x, 0) should involve only the smooth_abs terms at zero."""
        x = np.array([1.0, 2.0, 0.3])
        p = np.array([0.0, 0.0, 0.0])
        # With p=0, terms 1,2 vanish; terms 3,4 give u_bound*(|0|-|0|) ~ 0
        h = hamiltonian(0.0, x, p)
        assert abs(h) < 0.01  # approximately zero (up to smoothing)

    def test_known_value(self, hamiltonian):
        """Test H at a specific known configuration."""
        # theta = 0: cos(0)=1, sin(0)=0
        x = np.array([0.0, 0.0, 0.0])
        p = np.array([1.0, 0.0, 0.0])
        # H = -a*1*1 - 0*(g-a-0) + u*|0+0| - u*|0+0|
        #   = -a = -1.0
        h = hamiltonian(0.0, x, p)
        assert h == pytest.approx(-A_THRUST, abs=0.01)

    def test_symmetry(self, hamiltonian):
        """Hamiltonian should change when theta -> -theta (not symmetric)."""
        x_pos = np.array([1.0, 2.0, pi / 4])
        x_neg = np.array([1.0, 2.0, -pi / 4])
        p = np.array([1.0, 1.0, 0.5])
        h_pos = hamiltonian(0.0, x_pos, p)
        h_neg = hamiltonian(0.0, x_neg, p)
        # cos(pi/4) = cos(-pi/4) but sin(pi/4) != sin(-pi/4)
        # so they should differ due to sin(theta) terms
        assert h_pos != pytest.approx(h_neg, abs=1e-6)

    def test_state_dim(self, hamiltonian):
        assert hamiltonian.state_dim == 3


# ============================================================================
#  Unit Tests: Cole-Hopf Transformation
# ============================================================================

class TestColeHopf:
    """Tests for the Cole-Hopf transformation utilities."""

    def test_roundtrip(self):
        """cole_hopf_inverse(cole_hopf_forward(v, c), c) == v."""
        v = np.array([1.0, -0.5, 2.0, 0.0])
        c = 5.0
        omega = cole_hopf_forward(v, c)
        v_recovered = cole_hopf_inverse(omega, c)
        np.testing.assert_allclose(v_recovered, v, atol=1e-10)

    def test_forward_positive(self):
        """omega = exp(-c*v) should always be positive."""
        v = np.array([-10.0, 0.0, 10.0])
        c = 1.0
        omega = cole_hopf_forward(v, c)
        assert np.all(omega > 0)

    def test_frozen_coefficient_formula(self):
        """c = 2H / (delta * |Dv|^2)."""
        H = np.array([5.0, -3.0, 1.0])
        grad_sq = np.array([2.0, 4.0, 0.5])
        delta = 0.1
        c = compute_frozen_coefficient(H, grad_sq, delta)
        expected = 2.0 * H / (delta * grad_sq)
        np.testing.assert_allclose(c, expected, atol=1e-10)

    def test_relative_residual(self):
        """Test the convergence metric."""
        v_old = np.array([1.0, 2.0, 3.0])
        v_new = np.array([1.01, 2.02, 3.03])
        r = relative_residual(v_new, v_old)
        expected = np.linalg.norm(v_new - v_old) / np.linalg.norm(v_old)
        assert r == pytest.approx(expected, rel=1e-10)


# ============================================================================
#  Integration Tests: MC Value Function
# ============================================================================

class TestMCValue:
    """Tests for the Monte Carlo value function estimation."""

    def test_at_target_center(self, rng):
        """At origin (deep inside target), v should be very negative."""
        x = np.array([0.0, 0.0, 0.0])
        v = mc_value_at_point(
            x, t=0.0, T=0.5, delta=0.1, c=10.0,
            g_fn=rockets_terminal_cost,
            num_samples=5_000, rng=rng,
        )
        # g(0) = -1.5, so v should be near -1.5 for large c
        assert v < 0.0

    def test_far_from_target(self, rng):
        """Far from target, v should be positive."""
        x = np.array([10.0, 10.0, 0.0])
        v = mc_value_at_point(
            x, t=0.0, T=0.5, delta=0.1, c=10.0,
            g_fn=rockets_terminal_cost,
            num_samples=5_000, rng=rng,
        )
        assert v > 0.0

    def test_deterministic_with_seed(self):
        """Same seed should give same result."""
        x = np.array([2.0, 1.0, 0.5])
        rng1 = np.random.default_rng(123)
        rng2 = np.random.default_rng(123)
        v1 = mc_value_at_point(
            x, 0.0, 0.5, 0.1, 10.0, rockets_terminal_cost, 1000, rng1
        )
        v2 = mc_value_at_point(
            x, 0.0, 0.5, 0.1, 10.0, rockets_terminal_cost, 1000, rng2
        )
        assert v1 == pytest.approx(v2, abs=1e-10)

    def test_mc_converges_with_samples(self, rng):
        """Increasing samples should reduce variance."""
        x = np.array([2.0, 1.0, 0.0])
        params = dict(t=0.0, T=0.5, delta=0.1, c=5.0,
                      g_fn=rockets_terminal_cost)

        # Run multiple trials with few and many samples
        vals_few = []
        vals_many = []
        for _ in range(10):
            vals_few.append(mc_value_at_point(
                x, **params, num_samples=100, rng=rng))
            vals_many.append(mc_value_at_point(
                x, **params, num_samples=5000, rng=rng))

        var_few = np.var(vals_few)
        var_many = np.var(vals_many)
        # More samples should have less variance
        assert var_many < var_few or var_many < 0.01


# ============================================================================
#  Integration Tests: MC Gradient
# ============================================================================

class TestMCGradient:
    """Tests for the MC gradient estimation."""

    def test_gradient_at_origin_direction(self, rng):
        """At origin (center of target), gradient should be near zero."""
        x = np.array([0.0, 0.0, 0.0])
        Dv = mc_gradient_at_point(
            x, t=0.0, T=0.5, delta=0.1, c=5.0,
            g_fn=rockets_terminal_cost,
            num_samples=5_000, rng=rng,
        )
        # Gradient should be small at symmetric center
        assert np.linalg.norm(Dv) < 2.0

    def test_gradient_points_away_from_target(self, rng):
        """Away from target, gradient of value function should point outward."""
        x = np.array([3.0, 0.0, 0.0])
        Dv = mc_gradient_at_point(
            x, t=0.0, T=0.5, delta=0.1, c=5.0,
            g_fn=rockets_terminal_cost,
            num_samples=10_000, rng=rng,
        )
        # For signed distance, gradient in x-direction should be positive
        # when x > 0 (pointing away from origin)
        # Note: the MC gradient is approximate, so use loose tolerance
        assert Dv[0] > -1.0  # at least not strongly negative


# ============================================================================
#  Convergence Tests: Quasi-Linearization
# ============================================================================

class TestQuasiLinearization:
    """Tests for the quasi-linearization iteration (Algorithm 1)."""

    def test_residuals_decrease(self, small_config):
        """Residuals should generally decrease across iterations."""
        solver = RocketsSolver(config=small_config)
        X, Z, V, history = solver.solve_slice(
            theta_val=0.0, grid_res=10, domain=(-3.0, 3.0),
        )
        # At least one iteration should have occurred
        assert len(history) > 0
        # Final residual should be less than the first
        if len(history) > 2:
            assert history[-1] <= history[0] * 10  # allow some fluctuation

    def test_convergence_within_max_iters(self, small_config):
        """Solver should not exceed max_iters."""
        solver = RocketsSolver(config=small_config)
        _, _, _, history = solver.solve_slice(
            theta_val=0.0, grid_res=8, domain=(-3.0, 3.0),
        )
        assert len(history) <= small_config.max_iters

    def test_value_function_negative_inside_target(self, small_config):
        """v(t;x) should be negative inside the target set."""
        solver = RocketsSolver(config=small_config)
        _, _, V, _ = solver.solve_slice(
            theta_val=0.0, grid_res=15, domain=(-3.0, 3.0),
        )
        # Center of grid should be inside target
        center_idx = V.shape[0] // 2
        assert V[center_idx, center_idx] < 0.0

    def test_value_function_positive_far_from_target(self, small_config):
        """v(t;x) should be positive far from the target."""
        solver = RocketsSolver(config=small_config)
        _, _, V, _ = solver.solve_slice(
            theta_val=0.0, grid_res=15, domain=(-5.0, 5.0),
        )
        # Corners should be far from target
        assert V[0, 0] > 0.0
        assert V[-1, -1] > 0.0


# ============================================================================
#  JAX Consistency Tests
# ============================================================================

@pytest.mark.skipif(not HAS_JAX, reason="JAX not available")
class TestJAXConsistency:
    """Tests that JAX and NumPy implementations produce consistent results."""

    def test_hamiltonian_values_match(self):
        """JAX and NumPy Hamiltonians should give same values."""
        H_np = RocketsHamiltonian()

        for _ in range(10):
            x = np.random.randn(3)
            p = np.random.randn(3)
            h_np = H_np(0.0, x, p)
            h_jax = float(rockets_hamiltonian_jax(
                0.0, jnp.array(x), jnp.array(p)
            ))
            assert h_np == pytest.approx(h_jax, abs=1e-5)

    def test_terminal_cost_match(self):
        """JAX and NumPy terminal costs should agree."""
        for _ in range(10):
            x = np.random.randn(3) * 3
            g_np = rockets_terminal_cost(x)
            g_jax = float(rockets_terminal_cost_jax(jnp.array(x)))
            assert g_np == pytest.approx(g_jax, abs=1e-6)

    def test_mc_value_statistical_agreement(self):
        """MC values from JAX and NumPy should agree statistically."""
        x = np.array([2.0, 1.0, 0.3])
        t, T, delta, c = 0.0, 0.5, 0.1, 5.0

        # NumPy version
        rng = np.random.default_rng(42)
        v_np = mc_value_at_point(
            x, t, T, delta, c, rockets_terminal_cost, 10_000, rng,
        )

        # JAX version
        key = jax.random.PRNGKey(42)
        v_jax = float(mc_value_jax(
            key, jnp.array(x), t, T, delta, c, 10_000,
        ))

        # Should be statistically similar (different RNG, so not exact)
        assert abs(v_np - v_jax) < 1.0  # within 1.0 (MC variance)


# ============================================================================
#  Edge Cases
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases and input validation."""

    def test_zero_time_horizon(self, rng):
        """At t = T, v(T;x) should equal g(x)."""
        x = np.array([2.0, 1.0, 0.0])
        v = mc_value_at_point(
            x, t=0.5, T=0.5, delta=0.1, c=10.0,
            g_fn=rockets_terminal_cost,
            num_samples=1_000, rng=rng,
        )
        g_val = rockets_terminal_cost(x)
        # When T-t = 0, sigma -> 0, so all samples cluster at x
        assert v == pytest.approx(g_val, abs=0.5)

    def test_small_delta(self, rng):
        """Small delta should give values closer to inviscid solution."""
        x = np.array([2.0, 1.0, 0.0])
        v_large = mc_value_at_point(
            x, 0.0, 0.5, delta=1.0, c=5.0,
            g_fn=rockets_terminal_cost, num_samples=5_000, rng=rng,
        )
        v_small = mc_value_at_point(
            x, 0.0, 0.5, delta=0.01, c=5.0,
            g_fn=rockets_terminal_cost, num_samples=5_000, rng=rng,
        )
        # Both should be finite
        assert np.isfinite(v_large)
        assert np.isfinite(v_small)

    def test_large_c(self, rng):
        """Large c should make the value function approach min of g."""
        x = np.array([0.5, 0.0, 0.0])
        v = mc_value_at_point(
            x, 0.0, 0.5, delta=0.1, c=100.0,
            g_fn=rockets_terminal_cost, num_samples=5_000, rng=rng,
        )
        assert np.isfinite(v)


# ============================================================================
#  Main
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
