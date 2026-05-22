#!/usr/bin/env python
"""Rigorous tests for the Dubins car quasi-linearization solver.

Tests cover:
  1. Unit tests for Hamiltonians (simple and relative)
  2. Terminal cost function
  3. MC value/gradient estimation
  4. Quasi-linearization convergence
  5. JAX/NumPy consistency
  6. Edge cases

Run with: python -m pytest dubins_test.py -v
"""

import sys
import os
import numpy as np
import pytest
from math import pi

sys.path.insert(0, os.path.dirname(__file__))

from utils import (
    SolverConfig,
    cole_hopf_forward,
    cole_hopf_inverse,
    compute_frozen_coefficient,
    relative_residual,
    mc_value_at_point,
    mc_gradient_at_point,
    cylinder_cost,
)
from dubins_python import (
    DubinsHamiltonian,
    DubinsRelativeHamiltonian,
    dubins_terminal_cost,
    DubinsSolver,
)

try:
    import jax
    import jax.numpy as jnp
    from dubins_jax import (
        dubins_simple_hamiltonian,
        dubins_relative_hamiltonian,
        dubins_terminal_cost as dubins_terminal_cost_jax,
        DubinsSolverJAX,
    )
    HAS_JAX = True
except ImportError:
    HAS_JAX = False


# ============================================================================
#  Fixtures
# ============================================================================

@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def simple_H():
    return DubinsHamiltonian(speed=1.0, omega_max=1.0)


@pytest.fixture
def relative_H():
    return DubinsRelativeHamiltonian(v_p=-1.0, v_e=1.0, w=1.0)


@pytest.fixture
def small_config():
    return SolverConfig(
        delta=0.1,
        num_samples=2_000,
        max_iters=5,
        tol=1e-3,
        t_start=0.0,
        t_end=1.0,
        seed=42,
    )


# ============================================================================
#  Unit Tests: Terminal Cost
# ============================================================================

class TestDubinsTerminalCost:
    """Tests for g(x) = ||(x,y)|| - radius."""

    def test_at_origin(self):
        x = np.array([0.0, 0.0, 0.0])
        assert dubins_terminal_cost(x) == pytest.approx(-0.5, abs=1e-10)

    def test_on_boundary(self):
        r = 0.5
        for angle in np.linspace(0, 2 * pi, 8, endpoint=False):
            x = np.array([r * np.cos(angle), r * np.sin(angle), 0.0])
            assert dubins_terminal_cost(x) == pytest.approx(0.0, abs=1e-10)

    def test_outside(self):
        x = np.array([3.0, 4.0, 0.0])
        assert dubins_terminal_cost(x) > 0.0

    def test_inside(self):
        x = np.array([0.1, 0.1, 0.0])
        assert dubins_terminal_cost(x) < 0.0

    def test_theta_independence(self):
        x1 = np.array([1.0, 2.0, 0.0])
        x2 = np.array([1.0, 2.0, pi / 3])
        assert dubins_terminal_cost(x1) == pytest.approx(
            dubins_terminal_cost(x2), abs=1e-10)


# ============================================================================
#  Unit Tests: Simple Dubins Hamiltonian
# ============================================================================

class TestSimpleHamiltonian:
    """Tests for H(x,p) = v cos(theta) p1 + v sin(theta) p2 - omega_max |p3|."""

    def test_zero_costate(self, simple_H):
        x = np.array([0.0, 0.0, 0.5])
        p = np.array([0.0, 0.0, 0.0])
        h = simple_H(0.0, x, p)
        assert abs(h) < 0.01  # approximately zero

    def test_known_value_theta_zero(self, simple_H):
        """At theta=0: H = v*1*p1 + v*0*p2 - omega|p3|."""
        x = np.array([0.0, 0.0, 0.0])
        p = np.array([1.0, 0.0, 0.0])
        # H = 1*1*1 + 0 - 0 = 1.0
        assert simple_H(0.0, x, p) == pytest.approx(1.0, abs=0.01)

    def test_known_value_with_p3(self, simple_H):
        """With nonzero p3: omega_max * |p3| should subtract."""
        x = np.array([0.0, 0.0, 0.0])
        p = np.array([0.0, 0.0, 1.0])
        # H = 0 + 0 - 1*1 = -1.0
        assert simple_H(0.0, x, p) == pytest.approx(-1.0, abs=0.01)

    def test_state_dim(self, simple_H):
        assert simple_H.state_dim == 3


# ============================================================================
#  Unit Tests: Relative Dubins Hamiltonian
# ============================================================================

class TestRelativeHamiltonian:
    """Tests for the Merz (1972) pursuit-evasion Hamiltonian."""

    def test_zero_costate(self, relative_H):
        x = np.array([1.0, 2.0, 0.5])
        p = np.array([0.0, 0.0, 0.0])
        h = relative_H(0.0, x, p)
        assert abs(h) < 0.01

    def test_state_dim(self, relative_H):
        assert relative_H.state_dim == 3

    def test_finite_for_random_inputs(self, relative_H):
        """H should always return finite values."""
        for _ in range(20):
            x = np.random.randn(3)
            p = np.random.randn(3)
            h = relative_H(0.0, x, p)
            assert np.isfinite(h)


# ============================================================================
#  Integration Tests: MC Value
# ============================================================================

class TestDubinsMCValue:
    """Tests for MC value function on the Dubins problem."""

    def test_negative_inside_target(self, rng):
        x = np.array([0.0, 0.0, 0.0])
        v = mc_value_at_point(
            x, t=0.0, T=1.0, delta=0.1, c=10.0,
            g_fn=dubins_terminal_cost,
            num_samples=5_000, rng=rng,
        )
        assert v < 0.0

    def test_positive_far_from_target(self, rng):
        x = np.array([5.0, 5.0, 0.0])
        v = mc_value_at_point(
            x, t=0.0, T=1.0, delta=0.1, c=10.0,
            g_fn=dubins_terminal_cost,
            num_samples=5_000, rng=rng,
        )
        assert v > 0.0

    def test_finite_values(self, rng):
        for _ in range(5):
            x = np.random.randn(3) * 3
            v = mc_value_at_point(
                x, 0.0, 1.0, 0.1, 5.0,
                dubins_terminal_cost, 2_000, rng,
            )
            assert np.isfinite(v)


# ============================================================================
#  Convergence Tests
# ============================================================================

class TestDubinsQuasiLinearization:
    """Tests for convergence of Algorithm 1 on the Dubins problem."""

    def test_residuals_exist(self, small_config):
        solver = DubinsSolver(config=small_config, mode="pursuit_evasion")
        _, _, _, history = solver.solve_slice(
            theta_val=0.0, grid_res=8, domain=(-2.0, 2.0),
        )
        assert len(history) > 0

    def test_bounded_iters(self, small_config):
        solver = DubinsSolver(config=small_config, mode="pursuit_evasion")
        _, _, _, history = solver.solve_slice(
            theta_val=0.0, grid_res=8, domain=(-2.0, 2.0),
        )
        assert len(history) <= small_config.max_iters

    def test_negative_at_center(self, small_config):
        solver = DubinsSolver(config=small_config, mode="pursuit_evasion")
        _, _, V, _ = solver.solve_slice(
            theta_val=0.0, grid_res=15, domain=(-2.0, 2.0),
        )
        center = V.shape[0] // 2
        assert V[center, center] < 0.0

    def test_positive_at_corners(self, small_config):
        solver = DubinsSolver(config=small_config, mode="pursuit_evasion")
        _, _, V, _ = solver.solve_slice(
            theta_val=0.0, grid_res=15, domain=(-4.0, 4.0),
        )
        assert V[0, 0] > 0.0
        assert V[-1, -1] > 0.0

    def test_simple_mode_works(self, small_config):
        solver = DubinsSolver(config=small_config, mode="simple")
        _, _, V, history = solver.solve_slice(
            theta_val=0.0, grid_res=8, domain=(-2.0, 2.0),
        )
        assert V.shape == (8, 8)
        assert len(history) > 0


# ============================================================================
#  JAX Consistency
# ============================================================================

@pytest.mark.skipif(not HAS_JAX, reason="JAX not available")
class TestDubinsJAXConsistency:

    def test_simple_hamiltonian_match(self):
        H_np = DubinsHamiltonian()
        for _ in range(10):
            x = np.random.randn(3)
            p = np.random.randn(3)
            h_np = H_np(0.0, x, p)
            h_jax = float(dubins_simple_hamiltonian(
                0.0, jnp.array(x), jnp.array(p)
            ))
            assert h_np == pytest.approx(h_jax, abs=1e-5)

    def test_relative_hamiltonian_match(self):
        H_np = DubinsRelativeHamiltonian()
        for _ in range(10):
            x = np.random.randn(3)
            p = np.random.randn(3)
            h_np = H_np(0.0, x, p)
            h_jax = float(dubins_relative_hamiltonian(
                0.0, jnp.array(x), jnp.array(p)
            ))
            assert h_np == pytest.approx(h_jax, abs=1e-5)

    def test_terminal_cost_match(self):
        for _ in range(10):
            x = np.random.randn(3) * 3
            g_np = dubins_terminal_cost(x)
            g_jax = float(dubins_terminal_cost_jax(jnp.array(x)))
            assert g_np == pytest.approx(g_jax, abs=1e-6)


# ============================================================================
#  Edge Cases
# ============================================================================

class TestDubinsEdgeCases:

    def test_zero_time(self, rng):
        """At t=T, v should equal g."""
        x = np.array([1.0, 1.0, 0.0])
        v = mc_value_at_point(
            x, t=1.0, T=1.0, delta=0.1, c=10.0,
            g_fn=dubins_terminal_cost,
            num_samples=1_000, rng=rng,
        )
        g_val = dubins_terminal_cost(x)
        assert v == pytest.approx(g_val, abs=0.5)

    def test_various_theta_slices(self, small_config):
        """Solver should work for various theta values."""
        solver = DubinsSolver(config=small_config, mode="simple")
        for theta in [-pi, -pi/2, 0.0, pi/2, pi]:
            _, _, V, _ = solver.solve_slice(
                theta_val=theta, grid_res=6, domain=(-2.0, 2.0),
            )
            assert V.shape == (6, 6)
            assert np.all(np.isfinite(V))


# ============================================================================
#  Main
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
