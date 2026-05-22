"""Integration tests for the HJ reachability sampler."""

import jax
import jax.numpy as jnp
import pytest

from src.config import SolverConfig
from src.hj_sampler import HJReachabilitySampler
from src.hamiltonians.quadratic import QuadraticHamiltonian
from src.hamiltonians.double_integrator import DoubleIntegratorHamiltonian
from src.hamiltonians.rockets_relative import RocketsRelativeHamiltonian
from src.initial_conditions import sphere_cost, quadratic_cost


# ---------------------------------------------------------------------------
#  Exact Cole-Hopf tests (quadratic H)
# ---------------------------------------------------------------------------

class TestExactColeHopf:
    def setup_method(self):
        self.cfg = SolverConfig(
            delta=0.1, num_samples=20_000, seed=42,
            t_start=0.0, t_end=1.0, time_steps=5,
        )
        self.H = QuadraticHamiltonian(dim=2)

    def test_solve_returns_correct_shape(self):
        solver = HJReachabilitySampler(self.H, quadratic_cost, self.cfg)
        pts = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        v = solver.solve(pts, 0.5)
        assert v.shape == (3,)

    def test_value_finite(self):
        solver = HJReachabilitySampler(self.H, quadratic_cost, self.cfg)
        pts = jnp.array([[0.0, 0.0], [5.0, 5.0]])
        v = solver.solve(pts, 0.0)
        assert jnp.all(jnp.isfinite(v))

    def test_sphere_inside_vs_outside(self):
        """Inside the sphere: v < 0.  Outside: v > 0 (approximately)."""
        g = lambda x: sphere_cost(x, radius=2.0)
        solver = HJReachabilitySampler(self.H, g, self.cfg)
        inside = jnp.array([[0.0, 0.0]])
        outside = jnp.array([[5.0, 5.0]])
        v_in = solver.solve(inside, 0.9)   # close to terminal
        v_out = solver.solve(outside, 0.9)
        assert float(v_in[0]) < float(v_out[0])

    def test_reproducibility(self):
        """Same seed -> same result."""
        g = quadratic_cost
        pts = jnp.array([[1.0, 2.0]])
        s1 = HJReachabilitySampler(self.H, g, self.cfg)
        s2 = HJReachabilitySampler(self.H, g, self.cfg)
        v1 = s1.solve(pts, 0.5)
        v2 = s2.solve(pts, 0.5)
        assert jnp.allclose(v1, v2)


class TestBackwardSolve:
    def test_returns_dict(self):
        cfg = SolverConfig(
            delta=0.1, num_samples=5_000, seed=42,
            t_start=0.0, t_end=1.0, time_steps=3,
        )
        H = QuadraticHamiltonian(dim=2)
        solver = HJReachabilitySampler(H, quadratic_cost, cfg)
        pts = jnp.array([[0.0, 0.0], [1.0, 1.0]])
        result = solver.solve_backward(pts)
        assert "t" in result
        assert "v" in result
        assert result["v"].shape == (3, 2)  # (time_steps, M)


# ---------------------------------------------------------------------------
#  Quasi-linearization tests (general H)
# ---------------------------------------------------------------------------

class TestQuasiLinearDoubleIntegrator:
    """Test the quasi-linearization path with the double integrator."""

    def setup_method(self):
        self.cfg = SolverConfig(
            delta=0.1, num_samples=10_000, seed=42,
            max_quasi_iters=5, quasi_tol=1e-4,
            t_start=0.0, t_end=1.0, time_steps=3,
        )
        self.H = DoubleIntegratorHamiltonian(u_bound=1.0)
        self.g = lambda x: sphere_cost(x, radius=1.0)

    def test_returns_finite(self):
        solver = HJReachabilitySampler(self.H, self.g, self.cfg)
        pts = jnp.array([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]])
        v, history = solver.solve_quasi_linear(pts, 0.5)
        assert v.shape == (3,)
        assert jnp.all(jnp.isfinite(v))

    def test_history_non_empty(self):
        solver = HJReachabilitySampler(self.H, self.g, self.cfg)
        pts = jnp.array([[1.0, 1.0]])
        _, history = solver.solve_quasi_linear(pts, 0.5)
        assert len(history) >= 1

    def test_inside_vs_outside(self):
        """Value inside target < value outside (near terminal time)."""
        solver = HJReachabilitySampler(self.H, self.g, self.cfg)
        inside = jnp.array([[0.0, 0.0]])
        outside = jnp.array([[3.0, 3.0]])
        v_in, _ = solver.solve_quasi_linear(inside, 0.9)
        v_out, _ = solver.solve_quasi_linear(outside, 0.9)
        assert float(v_in[0]) < float(v_out[0])

    def test_solve_dispatches_to_quasi(self):
        """solve() should use quasi-linear for non-quadratic H."""
        solver = HJReachabilitySampler(self.H, self.g, self.cfg)
        pts = jnp.array([[0.0, 0.0]])
        v = solver.solve(pts, 0.5)
        assert v.shape == (1,)
        assert jnp.isfinite(v[0])


class TestQuasiLinearRockets:
    """Test quasi-linearization for the rockets system."""

    def setup_method(self):
        self.cfg = SolverConfig(
            delta=0.1, num_samples=5_000, seed=123,
            max_quasi_iters=3, quasi_tol=1e-3,
            t_start=0.0, t_end=1.0, time_steps=2,
        )
        self.H = RocketsRelativeHamiltonian(a=1.0, g=32.0, u_bound=1.0)
        from functools import partial
        from src.initial_conditions import cylinder_cost
        self.g = partial(cylinder_cost, axis_align=2, radius=1.5)

    def test_returns_finite(self):
        solver = HJReachabilitySampler(self.H, self.g, self.cfg)
        pts = jnp.array([
            [0.0, 0.0, 0.0],
            [5.0, 5.0, 0.5],
        ])
        v, history = solver.solve_quasi_linear(pts, 0.5)
        assert v.shape == (2,)
        assert jnp.all(jnp.isfinite(v))

    def test_inside_cylinder_lower(self):
        """Point inside the cylinder should have lower value than far away."""
        solver = HJReachabilitySampler(self.H, self.g, self.cfg)
        inside = jnp.array([[0.0, 0.0, 0.0]])
        outside = jnp.array([[10.0, 10.0, 0.0]])
        v_in, _ = solver.solve_quasi_linear(inside, 0.9)
        v_out, _ = solver.solve_quasi_linear(outside, 0.9)
        assert float(v_in[0]) < float(v_out[0])
