"""Load and soak tests for the HJ reachability sampler.

These tests verify robustness under stress conditions:
- Large sample counts (no OOM)
- Many quasi-linear iterations (no numerical drift)
- Repeated solves (variance consistency)
- Deterministic reproduction with fixed seeds
- Edge cases (tiny delta, large delta, zero time)
"""

import pytest
import jax
import jax.numpy as jnp

from src.config import SolverConfig
from src.hj_sampler import HJReachabilitySampler
from src.hamiltonians.quadratic import QuadraticHamiltonian
from src.hamiltonians.double_integrator import DoubleIntegratorHamiltonian
from src.initial_conditions import sphere_cost, quadratic_cost


# ---------------------------------------------------------------------------
#  Large sample count
# ---------------------------------------------------------------------------

class TestLargeSampleCount:
    """Verify no OOM or numerical issues with large J."""

    def test_100k_samples_exact(self):
        cfg = SolverConfig(
            delta=0.1, num_samples=100_000, seed=42,
            t_start=0.0, t_end=1.0, time_steps=3,
        )
        H = QuadraticHamiltonian(dim=2)
        solver = HJReachabilitySampler(H, quadratic_cost, cfg)
        pts = jnp.array([[0.0, 0.0], [1.0, 1.0]])
        v = solver.solve(pts, 0.5)
        assert v.shape == (2,)
        assert jnp.all(jnp.isfinite(v))

    def test_200k_samples_exact(self):
        cfg = SolverConfig(
            delta=0.1, num_samples=200_000, seed=42,
            t_start=0.0, t_end=1.0, time_steps=3,
        )
        H = QuadraticHamiltonian(dim=2)
        solver = HJReachabilitySampler(H, quadratic_cost, cfg)
        pts = jnp.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        v = solver.solve(pts, 0.5)
        assert v.shape == (3,)
        assert jnp.all(jnp.isfinite(v))

    def test_50k_samples_quasi_linear(self):
        cfg = SolverConfig(
            delta=0.1, num_samples=50_000, seed=42,
            t_start=0.0, t_end=1.0, time_steps=3,
            max_quasi_iters=3, quasi_tol=1e-3,
        )
        H = DoubleIntegratorHamiltonian(u_bound=1.0)
        g_fn = lambda x: sphere_cost(x, radius=1.0)
        solver = HJReachabilitySampler(H, g_fn, cfg)
        pts = jnp.array([[0.0, 0.0], [2.0, 0.0]])
        v, history = solver.solve_quasi_linear(pts, 0.5)
        assert v.shape == (2,)
        assert jnp.all(jnp.isfinite(v))


# ---------------------------------------------------------------------------
#  Many quasi-linear iterations (no drift)
# ---------------------------------------------------------------------------

class TestManyIterations:
    """Verify stability over many quasi-linear iterations."""

    def test_20_iterations_stable(self):
        cfg = SolverConfig(
            delta=0.1, num_samples=10_000, seed=42,
            t_start=0.0, t_end=1.0, time_steps=3,
            max_quasi_iters=20, quasi_tol=1e-10,  # won't converge, force 20 iters
        )
        H = DoubleIntegratorHamiltonian(u_bound=1.0)
        g_fn = lambda x: sphere_cost(x, radius=1.0)
        solver = HJReachabilitySampler(H, g_fn, cfg)
        pts = jnp.array([[0.0, 0.0], [1.5, 0.0]])
        v, history = solver.solve_quasi_linear(pts, 0.5)
        assert jnp.all(jnp.isfinite(v))
        # No NaN or Inf should appear at any iteration
        assert all(jnp.isfinite(jnp.array(h)) for h in history)

    def test_residuals_dont_explode(self):
        """Residuals should not grow unboundedly."""
        cfg = SolverConfig(
            delta=0.1, num_samples=10_000, seed=42,
            t_start=0.0, t_end=1.0, time_steps=3,
            max_quasi_iters=15, quasi_tol=1e-10,
        )
        H = DoubleIntegratorHamiltonian(u_bound=1.0)
        g_fn = lambda x: sphere_cost(x, radius=1.0)
        solver = HJReachabilitySampler(H, g_fn, cfg)
        pts = jnp.array([[1.0, 1.0]])
        _, history = solver.solve_quasi_linear(pts, 0.5)
        # Last residual should not be much larger than first
        assert history[-1] < 100 * max(history[0], 1e-6)


# ---------------------------------------------------------------------------
#  Repeated solves (variance consistency)
# ---------------------------------------------------------------------------

class TestRepeatedSolves:
    """Verify variance is consistent across repeated independent solves."""

    def test_variance_bounded(self):
        """100 solves with different seeds should have bounded spread."""
        H = QuadraticHamiltonian(dim=2)
        pt = jnp.array([[1.0, 0.5]])
        results = []
        for seed in range(50):
            cfg = SolverConfig(
                delta=0.1, num_samples=10_000, seed=seed,
                t_start=0.0, t_end=1.0, time_steps=3,
            )
            solver = HJReachabilitySampler(H, quadratic_cost, cfg)
            v = solver.solve(pt, 0.5)
            results.append(float(v[0]))

        import numpy as np
        results = np.array(results)
        std = results.std()
        # For J=10k, std should be O(1/sqrt(J)) ~ 0.01
        assert std < 0.1, f"Variance too high: std={std:.4f}"
        # All results should be finite
        assert np.all(np.isfinite(results))

    def test_same_seed_reproducible(self):
        """Same seed must give identical results."""
        H = QuadraticHamiltonian(dim=2)
        pts = jnp.array([[1.0, 2.0], [0.0, 0.0]])
        cfg = SolverConfig(
            delta=0.1, num_samples=10_000, seed=42,
            t_start=0.0, t_end=1.0, time_steps=3,
        )
        s1 = HJReachabilitySampler(H, quadratic_cost, cfg)
        s2 = HJReachabilitySampler(H, quadratic_cost, cfg)
        v1 = s1.solve(pts, 0.5)
        v2 = s2.solve(pts, 0.5)
        assert jnp.allclose(v1, v2, atol=1e-6)


# ---------------------------------------------------------------------------
#  Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Edge-case robustness."""

    def test_tiny_delta(self):
        """Very small viscosity: should still be finite."""
        cfg = SolverConfig(
            delta=0.001, num_samples=10_000, seed=42,
            t_start=0.0, t_end=1.0, time_steps=3,
        )
        H = QuadraticHamiltonian(dim=2)
        solver = HJReachabilitySampler(H, quadratic_cost, cfg)
        pts = jnp.array([[0.0, 0.0], [0.5, 0.5]])
        v = solver.solve(pts, 0.5)
        assert jnp.all(jnp.isfinite(v))

    def test_large_delta(self):
        """Large viscosity: heavy smoothing, should still be finite."""
        cfg = SolverConfig(
            delta=10.0, num_samples=10_000, seed=42,
            t_start=0.0, t_end=1.0, time_steps=3,
        )
        H = QuadraticHamiltonian(dim=2)
        solver = HJReachabilitySampler(H, quadratic_cost, cfg)
        pts = jnp.array([[0.0, 0.0], [1.0, 1.0]])
        v = solver.solve(pts, 0.5)
        assert jnp.all(jnp.isfinite(v))

    def test_near_terminal_time(self):
        """t very close to T: sigma ~0, v ~ g(x)."""
        cfg = SolverConfig(
            delta=0.1, num_samples=10_000, seed=42,
            t_start=0.0, t_end=1.0, time_steps=3,
        )
        H = QuadraticHamiltonian(dim=2)
        solver = HJReachabilitySampler(H, quadratic_cost, cfg)
        pts = jnp.array([[1.0, 1.0]])
        v = solver.solve(pts, 0.999)
        # Should be very close to g(x) = |x|^2 = 2.0
        assert abs(float(v[0]) - 2.0) < 0.1

    def test_single_point(self):
        """Single evaluation point."""
        cfg = SolverConfig(
            delta=0.1, num_samples=5_000, seed=42,
            t_start=0.0, t_end=1.0, time_steps=3,
        )
        H = QuadraticHamiltonian(dim=2)
        solver = HJReachabilitySampler(H, quadratic_cost, cfg)
        pt = jnp.array([[3.0, 4.0]])
        v = solver.solve(pt, 0.5)
        assert v.shape == (1,)
        assert jnp.isfinite(v[0])

    def test_many_points(self):
        """Large batch of evaluation points."""
        cfg = SolverConfig(
            delta=0.1, num_samples=5_000, seed=42,
            t_start=0.0, t_end=1.0, time_steps=3,
        )
        H = QuadraticHamiltonian(dim=2)
        solver = HJReachabilitySampler(H, quadratic_cost, cfg)
        pts = jax.random.normal(jax.random.PRNGKey(0), shape=(100, 2))
        v = solver.solve(pts, 0.5)
        assert v.shape == (100,)
        assert jnp.all(jnp.isfinite(v))

    def test_higher_dimension(self):
        """3D state space (e.g. rockets-like)."""
        cfg = SolverConfig(
            delta=0.1, num_samples=10_000, seed=42,
            t_start=0.0, t_end=1.0, time_steps=3,
        )
        H = QuadraticHamiltonian(dim=3)
        g_fn = lambda x: jnp.sum(x ** 2)
        solver = HJReachabilitySampler(H, g_fn, cfg)
        pts = jnp.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        v = solver.solve(pts, 0.5)
        assert v.shape == (2,)
        assert jnp.all(jnp.isfinite(v))


# ---------------------------------------------------------------------------
#  Backward solve soak
# ---------------------------------------------------------------------------

class TestBackwardSolveSoak:
    """Verify backward solve over many time steps."""

    def test_many_time_steps(self):
        cfg = SolverConfig(
            delta=0.1, num_samples=5_000, seed=42,
            t_start=0.0, t_end=1.0, time_steps=20,
        )
        H = QuadraticHamiltonian(dim=2)
        solver = HJReachabilitySampler(H, quadratic_cost, cfg)
        pts = jnp.array([[0.0, 0.0], [1.0, 1.0]])
        result = solver.solve_backward(pts)
        assert result["v"].shape == (20, 2)
        assert jnp.all(jnp.isfinite(result["v"]))

    def test_backward_monotonicity_at_origin(self):
        """At the origin with sphere cost, value should stay negative
        (inside) across all backward time steps."""
        cfg = SolverConfig(
            delta=0.1, num_samples=10_000, seed=42,
            t_start=0.0, t_end=1.0, time_steps=10,
        )
        H = QuadraticHamiltonian(dim=2)
        g_fn = lambda x: sphere_cost(x, radius=2.0)
        solver = HJReachabilitySampler(H, g_fn, cfg)
        pts = jnp.array([[0.0, 0.0]])
        result = solver.solve_backward(pts)
        # Origin is well inside sphere; value should be negative at all times
        assert jnp.all(result["v"][:, 0] < 0)
