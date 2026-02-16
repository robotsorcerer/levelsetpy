"""
Comprehensive tests for spatial derivative approximations.
Tests: upwindFirstFirst, upwindFirstENO2, upwindFirstENO3a, upwindFirstWENO5a
"""

import pytest
import torch
import numpy as np
from math import pi

from levelsetpy.grids import createGrid
from levelsetpy.utilities import Bundle
from levelsetpy.boundarycondition import addGhostPeriodic
from levelsetpy.spatialderivative import (
    upwindFirstFirst,
    upwindFirstENO2,
    upwindFirstENO3a,
    upwindFirstENO3b,
    upwindFirstWENO5a,
    upwindFirstWENO5b,
)
from levelsetpy.spatialderivative.upwind_first_eno3 import upwindFirstENO3
from levelsetpy.spatialderivative.upwind_first_weno5 import upwindFirstWENO5


class TestUpwindFirstFirst:
    """Tests for first-order upwind derivative."""

    def test_returns_two_arrays(self, grid_2d_periodic, linear_data_2d):
        """Returns (derivL, derivR) tuple."""
        derivL, derivR = upwindFirstFirst(grid_2d_periodic, linear_data_2d, 0)
        assert isinstance(derivL, torch.Tensor)
        assert isinstance(derivR, torch.Tensor)

    def test_output_shape_matches_input(self, grid_2d_periodic, linear_data_2d):
        """Derivatives have same shape as input data."""
        for dim in range(2):
            derivL, derivR = upwindFirstFirst(grid_2d_periodic, linear_data_2d, dim)
            assert derivL.shape == linear_data_2d.shape, f"derivL shape mismatch dim={dim}"
            assert derivR.shape == linear_data_2d.shape, f"derivR shape mismatch dim={dim}"

    def test_linear_function_exact_dim0(self, grid_2d_periodic):
        """For f(x,y)=2x+3y, df/dx should be exactly 2."""
        data = 2.0 * grid_2d_periodic.xs[0] + 3.0 * grid_2d_periodic.xs[1]
        data = data.to(torch.float64)
        derivL, derivR = upwindFirstFirst(grid_2d_periodic, data, 0)
        # Interior should be close to 2.0 (boundary effects possible)
        interior = derivL[2:-2, 2:-2]
        assert torch.allclose(interior, torch.full_like(interior, 2.0), atol=0.1)

    def test_linear_function_exact_dim1(self, grid_2d_periodic):
        """For f(x,y)=2x+3y, df/dy should be exactly 3."""
        data = 2.0 * grid_2d_periodic.xs[0] + 3.0 * grid_2d_periodic.xs[1]
        data = data.to(torch.float64)
        derivL, derivR = upwindFirstFirst(grid_2d_periodic, data, 1)
        interior = derivL[2:-2, 2:-2]
        assert torch.allclose(interior, torch.full_like(interior, 3.0), atol=0.1)

    def test_finite_output(self, grid_2d_periodic, sample_data_2d):
        """No NaN or Inf."""
        for dim in range(2):
            derivL, derivR = upwindFirstFirst(grid_2d_periodic, sample_data_2d, dim)
            assert torch.isfinite(derivL).all(), f"NaN/Inf in derivL dim={dim}"
            assert torch.isfinite(derivR).all(), f"NaN/Inf in derivR dim={dim}"

    def test_invalid_dim_raises(self, grid_2d_periodic, sample_data_2d):
        """Invalid dimension raises ValueError."""
        with pytest.raises(ValueError):
            upwindFirstFirst(grid_2d_periodic, sample_data_2d, -1)


class TestUpwindFirstENO2:
    """Tests for second-order ENO derivative."""

    def test_returns_two_arrays(self, grid_2d_periodic, linear_data_2d):
        derivL, derivR = upwindFirstENO2(grid_2d_periodic, linear_data_2d, 0)
        assert isinstance(derivL, torch.Tensor)
        assert isinstance(derivR, torch.Tensor)

    def test_output_shape(self, grid_2d_periodic, linear_data_2d):
        for dim in range(2):
            derivL, derivR = upwindFirstENO2(grid_2d_periodic, linear_data_2d, dim)
            assert derivL.shape == linear_data_2d.shape
            assert derivR.shape == linear_data_2d.shape

    def test_linear_exact_dim0(self, grid_2d_periodic):
        """ENO2 should be exact for linear functions."""
        data = 2.0 * grid_2d_periodic.xs[0] + 3.0 * grid_2d_periodic.xs[1]
        data = data.to(torch.float64)
        derivL, derivR = upwindFirstENO2(grid_2d_periodic, data, 0)
        interior = derivL[3:-3, 3:-3]
        assert torch.allclose(interior, torch.full_like(interior, 2.0), atol=0.05)

    def test_quadratic_convergence(self, grid_2d_periodic):
        """ENO2 is second order -- error decreases with h^2."""
        # This is a convergence rate test
        errors = []
        for N_val in [21, 41, 81]:
            gmin = -1.0 * np.ones((2, 1), dtype=np.float64)
            gmax = +1.0 * np.ones((2, 1), dtype=np.float64)
            N = N_val * np.ones((2, 1), dtype=np.int64)
            g = createGrid(gmin, gmax, N, process=True)
            g.xs = [torch.as_tensor(x) for x in g.xs]
            # Use a quadratic: x^2, derivative = 2x
            data = (g.xs[0] ** 2).to(torch.float64)
            derivL, derivR = upwindFirstENO2(g, data, 0)
            exact = 2.0 * g.xs[0].to(torch.float64)
            # Average of left and right
            centered = 0.5 * (derivL + derivR)
            err = torch.abs(centered[4:-4, 4:-4] - exact[4:-4, 4:-4]).max().item()
            errors.append(err)
        # Check convergence: error ratio should be ~4 for 2nd order
        if errors[0] > 1e-10:  # avoid division issues
            ratio = errors[0] / errors[1]
            assert ratio > 2.0, f"Expected ~4x error reduction, got {ratio:.2f}"

    def test_finite_output(self, grid_2d_periodic, sample_data_2d):
        for dim in range(2):
            derivL, derivR = upwindFirstENO2(grid_2d_periodic, sample_data_2d, dim)
            assert torch.isfinite(derivL).all()
            assert torch.isfinite(derivR).all()

    def test_generate_all_mode(self, grid_2d_periodic, sample_data_2d):
        """generateAll=True returns lists of approximations."""
        derivL, derivR = upwindFirstENO2(grid_2d_periodic, sample_data_2d, 0, generateAll=True)
        assert isinstance(derivL, list)
        assert isinstance(derivR, list)
        assert len(derivL) == 2
        assert len(derivR) == 2


class TestUpwindFirstENO3a:
    """Tests for third-order ENO derivative."""

    def test_returns_two_arrays(self, grid_2d_periodic, linear_data_2d):
        derivL, derivR = upwindFirstENO3a(grid_2d_periodic, linear_data_2d, 0)
        assert isinstance(derivL, torch.Tensor)
        assert isinstance(derivR, torch.Tensor)

    def test_output_shape(self, grid_2d_periodic, linear_data_2d):
        for dim in range(2):
            derivL, derivR = upwindFirstENO3a(grid_2d_periodic, linear_data_2d, dim)
            assert derivL.shape == linear_data_2d.shape
            assert derivR.shape == linear_data_2d.shape

    def test_linear_exact(self, grid_2d_periodic):
        """ENO3 exact on linear functions."""
        data = 2.0 * grid_2d_periodic.xs[0] + 3.0 * grid_2d_periodic.xs[1]
        data = data.to(torch.float64)
        derivL, derivR = upwindFirstENO3a(grid_2d_periodic, data, 0)
        interior = derivL[5:-5, 5:-5]
        assert torch.allclose(interior, torch.full_like(interior, 2.0), atol=0.05)

    def test_finite_output(self, grid_2d_periodic, sample_data_2d):
        for dim in range(2):
            derivL, derivR = upwindFirstENO3a(grid_2d_periodic, sample_data_2d, dim)
            assert torch.isfinite(derivL).all()
            assert torch.isfinite(derivR).all()


class TestUpwindFirstENO3b:
    """Tests for third-order ENO derivative (direct computation variant)."""

    def test_returns_two_arrays(self, grid_2d_periodic, linear_data_2d):
        derivL, derivR = upwindFirstENO3b(grid_2d_periodic, linear_data_2d, 0)
        assert isinstance(derivL, torch.Tensor)
        assert isinstance(derivR, torch.Tensor)

    def test_output_shape(self, grid_2d_periodic, linear_data_2d):
        for dim in range(2):
            derivL, derivR = upwindFirstENO3b(grid_2d_periodic, linear_data_2d, dim)
            assert derivL.shape == linear_data_2d.shape
            assert derivR.shape == linear_data_2d.shape

    def test_linear_exact(self, grid_2d_periodic):
        """ENO3b exact on linear functions."""
        data = 2.0 * grid_2d_periodic.xs[0] + 3.0 * grid_2d_periodic.xs[1]
        data = data.to(torch.float64)
        derivL, derivR = upwindFirstENO3b(grid_2d_periodic, data, 0)
        interior = derivL[5:-5, 5:-5]
        assert torch.allclose(interior, torch.full_like(interior, 2.0), atol=0.05)

    def test_finite_output(self, grid_2d_periodic, sample_data_2d):
        for dim in range(2):
            derivL, derivR = upwindFirstENO3b(grid_2d_periodic, sample_data_2d, dim)
            assert torch.isfinite(derivL).all()
            assert torch.isfinite(derivR).all()

    def test_agrees_with_eno3a(self, grid_2d_periodic):
        """ENO3b and ENO3a should agree on smooth data (interior points)."""
        data = 2.0 * grid_2d_periodic.xs[0] + 3.0 * grid_2d_periodic.xs[1]
        data = data.to(torch.float64)
        La, Ra = upwindFirstENO3a(grid_2d_periodic, data, 0)
        Lb, Rb = upwindFirstENO3b(grid_2d_periodic, data, 0)
        # Skip boundary rows where periodic wrapping of the linear function
        # creates a discontinuity — ENO stencil selection may legitimately differ.
        s = 3  # stencil width
        assert torch.allclose(La[s:-s, :], Lb[s:-s, :], atol=1e-10)
        assert torch.allclose(Ra[s:-s, :], Rb[s:-s, :], atol=1e-10)

    def test_generate_all_mode(self, grid_2d_periodic, sample_data_2d):
        """generateAll=True returns lists of 3 ENO approximations."""
        derivL, derivR = upwindFirstENO3b(grid_2d_periodic, sample_data_2d, 0, generateAll=True)
        assert isinstance(derivL, list)
        assert isinstance(derivR, list)
        assert len(derivL) == 3
        assert len(derivR) == 3


class TestWrapperAliases:
    """Tests for wrapper modules that delegate to a/b variants."""

    def test_eno3_delegates_to_eno3a(self, grid_2d_periodic):
        """upwindFirstENO3 wraps upwindFirstENO3a."""
        data = 2.0 * grid_2d_periodic.xs[0] + 3.0 * grid_2d_periodic.xs[1]
        data = data.to(torch.float64)
        La, Ra = upwindFirstENO3a(grid_2d_periodic, data, 0)
        L, R = upwindFirstENO3(grid_2d_periodic, data, 0)
        assert torch.equal(La, L)
        assert torch.equal(Ra, R)

    def test_weno5_delegates_to_weno5a(self, grid_2d_periodic):
        """upwindFirstWENO5 wraps upwindFirstWENO5a."""
        data = 2.0 * grid_2d_periodic.xs[0] + 3.0 * grid_2d_periodic.xs[1]
        data = data.to(torch.float64)
        La, Ra = upwindFirstWENO5a(grid_2d_periodic, data, 0)
        L, R = upwindFirstWENO5(grid_2d_periodic, data, 0)
        assert torch.equal(La, L)
        assert torch.equal(Ra, R)


class TestUpwindFirstWENO5a:
    """Tests for fifth-order WENO derivative."""

    def test_returns_two_arrays(self, grid_2d_periodic, linear_data_2d):
        derivL, derivR = upwindFirstWENO5a(grid_2d_periodic, linear_data_2d, 0)
        assert isinstance(derivL, torch.Tensor)
        assert isinstance(derivR, torch.Tensor)

    def test_output_shape(self, grid_2d_periodic, linear_data_2d):
        for dim in range(2):
            derivL, derivR = upwindFirstWENO5a(grid_2d_periodic, linear_data_2d, dim)
            assert derivL.shape == linear_data_2d.shape
            assert derivR.shape == linear_data_2d.shape

    def test_linear_exact(self, grid_2d_periodic):
        """WENO5 exact on linear functions."""
        data = 2.0 * grid_2d_periodic.xs[0] + 3.0 * grid_2d_periodic.xs[1]
        data = data.to(torch.float64)
        derivL, derivR = upwindFirstWENO5a(grid_2d_periodic, data, 0)
        interior = derivL[5:-5, 5:-5]
        assert torch.allclose(interior, torch.full_like(interior, 2.0), atol=0.1)

    def test_finite_output(self, grid_2d_periodic, sample_data_2d):
        for dim in range(2):
            derivL, derivR = upwindFirstWENO5a(grid_2d_periodic, sample_data_2d, dim)
            assert torch.isfinite(derivL).all()
            assert torch.isfinite(derivR).all()


class TestUpwindFirstWENO5b:
    """Tests for fifth-order WENO derivative (direct computation variant)."""

    def test_returns_two_arrays(self, grid_2d_periodic, linear_data_2d):
        derivL, derivR = upwindFirstWENO5b(grid_2d_periodic, linear_data_2d, 0)
        assert isinstance(derivL, torch.Tensor)
        assert isinstance(derivR, torch.Tensor)

    def test_output_shape(self, grid_2d_periodic, linear_data_2d):
        for dim in range(2):
            derivL, derivR = upwindFirstWENO5b(grid_2d_periodic, linear_data_2d, dim)
            assert derivL.shape == linear_data_2d.shape
            assert derivR.shape == linear_data_2d.shape

    def test_linear_exact(self, grid_2d_periodic):
        """WENO5b exact on linear functions."""
        data = 2.0 * grid_2d_periodic.xs[0] + 3.0 * grid_2d_periodic.xs[1]
        data = data.to(torch.float64)
        derivL, derivR = upwindFirstWENO5b(grid_2d_periodic, data, 0)
        interior = derivL[5:-5, 5:-5]
        assert torch.allclose(interior, torch.full_like(interior, 2.0), atol=0.1)

    def test_finite_output(self, grid_2d_periodic, sample_data_2d):
        for dim in range(2):
            derivL, derivR = upwindFirstWENO5b(grid_2d_periodic, sample_data_2d, dim)
            assert torch.isfinite(derivL).all()
            assert torch.isfinite(derivR).all()

    def test_agrees_with_weno5a(self, grid_2d_periodic):
        """WENO5b and WENO5a should agree on smooth data."""
        data = 2.0 * grid_2d_periodic.xs[0] + 3.0 * grid_2d_periodic.xs[1]
        data = data.to(torch.float64)
        La, Ra = upwindFirstWENO5a(grid_2d_periodic, data, 0)
        Lb, Rb = upwindFirstWENO5b(grid_2d_periodic, data, 0)
        assert torch.allclose(La, Lb, atol=1e-10), f"L max diff: {torch.abs(La - Lb).max()}"
        assert torch.allclose(Ra, Rb, atol=1e-10), f"R max diff: {torch.abs(Ra - Rb).max()}"

    def test_generate_all_mode(self, grid_2d_periodic, sample_data_2d):
        """generateAll=True returns lists of 3 ENO approximations."""
        derivL, derivR = upwindFirstWENO5b(grid_2d_periodic, sample_data_2d, 0, generateAll=True)
        assert isinstance(derivL, list)
        assert isinstance(derivR, list)
        assert len(derivL) == 3
        assert len(derivR) == 3


class TestCrossSchemeConsistency:
    """Cross-scheme checks for derivative methods."""

    def test_working_schemes_same_shape(self, grid_2d_periodic, sample_data_2d):
        """All schemes produce same output shape."""
        schemes = [upwindFirstFirst, upwindFirstENO2, upwindFirstENO3a, upwindFirstENO3b, upwindFirstWENO5a, upwindFirstWENO5b]
        for dim in range(2):
            shapes = []
            for scheme in schemes:
                dL, dR = scheme(grid_2d_periodic, sample_data_2d, dim)
                shapes.append(dL.shape)
            assert all(s == shapes[0] for s in shapes), f"Shape mismatch dim={dim}: {shapes}"

    def test_linear_all_agree(self, grid_2d_periodic):
        """Working schemes agree on linear function derivative."""
        data = 2.0 * grid_2d_periodic.xs[0] + 3.0 * grid_2d_periodic.xs[1]
        data = data.to(torch.float64)
        schemes = [upwindFirstFirst, upwindFirstENO2, upwindFirstENO3a, upwindFirstENO3b, upwindFirstWENO5a, upwindFirstWENO5b]
        results = []
        for scheme in schemes:
            dL, dR = scheme(grid_2d_periodic, data, 0)
            centered = 0.5 * (dL + dR)
            results.append(centered[5:-5, 5:-5])
        # All should be close to 2.0
        for i, r in enumerate(results):
            assert torch.allclose(r, torch.full_like(r, 2.0), atol=0.2), f"Scheme {i} failed"


class TestDeterminism:
    """Verify deterministic computation."""

    def test_repeated_calls_identical(self, grid_2d_periodic, sample_data_2d):
        """Same inputs produce bitwise identical outputs."""
        dL1, dR1 = upwindFirstENO2(grid_2d_periodic, sample_data_2d, 0)
        dL2, dR2 = upwindFirstENO2(grid_2d_periodic, sample_data_2d, 0)
        assert torch.equal(dL1, dL2)
        assert torch.equal(dR1, dR2)
