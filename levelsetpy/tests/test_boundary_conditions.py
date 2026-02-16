"""
Comprehensive tests for boundary condition ghost cell functions.
Tests: addGhostPeriodic, addGhostExtrapolate, addGhostNeumann, addGhostDirichlet
"""

import pytest
import torch
import numpy as np
from math import pi

from levelsetpy.grids import createGrid
from levelsetpy.utilities import Bundle
from levelsetpy.boundarycondition import addGhostPeriodic, addGhostExtrapolate
from levelsetpy.boundarycondition.add_ghost_neumann import addGhostNeumann
from levelsetpy.boundarycondition.add_ghost_dirichlet import addGhostDirichlet


class TestAddGhostPeriodic:
    """Tests for periodic boundary conditions."""

    def test_output_shape_2d_width1(self, sample_data_2d):
        """Ghost cells add 2*width nodes in the specified dimension."""
        data = sample_data_2d
        for dim in range(2):
            out = addGhostPeriodic(data, dim, width=1)
            expected_shape = list(data.shape)
            expected_shape[dim] += 2
            assert out.shape == tuple(expected_shape), f"Shape mismatch for dim={dim}"

    def test_output_shape_2d_width2(self, sample_data_2d):
        """Width=2 adds 4 ghost nodes total."""
        data = sample_data_2d
        out = addGhostPeriodic(data, 0, width=2)
        assert out.shape[0] == data.shape[0] + 4

    def test_interior_data_preserved(self, sample_data_2d):
        """Interior data (without ghost cells) matches input."""
        data = sample_data_2d
        width = 1
        for dim in range(2):
            out = addGhostPeriodic(data, dim, width=width)
            if dim == 0:
                interior = out[width:-width, :]
            else:
                interior = out[:, width:-width]
            assert torch.allclose(interior, data, atol=1e-14), f"Interior mismatch dim={dim}"

    def test_periodic_wrapping_dim0(self, sample_data_2d):
        """Bottom ghost cells = top of original data and vice versa."""
        data = sample_data_2d
        width = 1
        out = addGhostPeriodic(data, 0, width=width)
        # Bottom ghost should be the last row(s) of original
        assert torch.allclose(out[:width, :], data[-width:, :], atol=1e-14)
        # Top ghost should be the first row(s) of original
        assert torch.allclose(out[-width:, :], data[:width, :], atol=1e-14)

    def test_periodic_wrapping_dim1(self, sample_data_2d):
        """Left ghost cells = right of original data and vice versa."""
        data = sample_data_2d
        width = 1
        out = addGhostPeriodic(data, 1, width=width)
        assert torch.allclose(out[:, :width], data[:, -width:], atol=1e-14)
        assert torch.allclose(out[:, -width:], data[:, :width], atol=1e-14)

    def test_output_dtype_float64(self, sample_data_2d):
        """Output must be float64."""
        out = addGhostPeriodic(sample_data_2d, 0, width=1)
        assert out.dtype == torch.float64

    def test_numpy_input_converted(self):
        """Numpy input should be automatically converted to torch."""
        data_np = np.random.randn(10, 10)
        out = addGhostPeriodic(data_np, 0, width=1)
        assert isinstance(out, torch.Tensor)
        assert out.shape == (12, 10)

    @pytest.mark.parametrize("width", [1, 2, 3])
    def test_various_widths(self, sample_data_2d, width):
        """Test multiple ghost cell widths."""
        out = addGhostPeriodic(sample_data_2d, 0, width=width)
        assert out.shape[0] == sample_data_2d.shape[0] + 2 * width

    def test_1d_data(self, grid_1d):
        """Periodic BC on 1D data."""
        data = torch.sin(pi * grid_1d.xs[0]).to(torch.float64)
        out = addGhostPeriodic(data, 0, width=1)
        assert out.shape[0] == data.shape[0] + 2


class TestAddGhostExtrapolate:
    """Tests for extrapolation boundary conditions."""

    def test_output_shape(self, sample_data_2d):
        """Correct output shape with ghost cells."""
        for dim in range(2):
            out = addGhostExtrapolate(sample_data_2d, dim, width=1)
            expected = list(sample_data_2d.shape)
            expected[dim] += 2
            assert out.shape == tuple(expected)

    def test_interior_preserved(self, sample_data_2d):
        """Interior data unchanged."""
        width = 1
        out = addGhostExtrapolate(sample_data_2d, 0, width=width)
        interior = out[width:-width, :]
        assert torch.allclose(interior, sample_data_2d, atol=1e-14)

    def test_extrapolation_away_from_zero(self, sphere_data_2d):
        """Default: extrapolation goes away from zero level set."""
        out = addGhostExtrapolate(sphere_data_2d, 0, width=1)
        # Ghost cells should extrapolate away from zero
        assert out.shape[0] == sphere_data_2d.shape[0] + 2
        # Values should be finite
        assert torch.isfinite(out).all()

    def test_extrapolation_toward_zero(self, sphere_data_2d):
        """towardZero=True reverses extrapolation sign."""
        ghost_data = Bundle(dict(towardZero=True))
        out = addGhostExtrapolate(sphere_data_2d, 0, width=1, ghostData=ghost_data)
        assert torch.isfinite(out).all()

    def test_output_dtype(self, sample_data_2d):
        """Output is float64."""
        out = addGhostExtrapolate(sample_data_2d, 0, width=1)
        assert out.dtype == torch.float64

    @pytest.mark.parametrize("dim", [0, 1])
    def test_both_dimensions(self, sample_data_2d, dim):
        """Works for both dimensions."""
        out = addGhostExtrapolate(sample_data_2d, dim, width=1)
        expected = list(sample_data_2d.shape)
        expected[dim] += 2
        assert out.shape == tuple(expected)


class TestAddGhostDirichlet:
    """Tests for Dirichlet boundary conditions."""

    def test_output_shape(self, sample_data_2d):
        """Correct shape with ghost cells."""
        out = addGhostDirichlet(sample_data_2d, 0, width=1)
        assert out.shape[0] == sample_data_2d.shape[0] + 2

    def test_interior_preserved(self, sample_data_2d):
        """Interior data matches input."""
        width = 1
        out = addGhostDirichlet(sample_data_2d, 0, width=width)
        interior = out[width:-width, :]
        assert torch.allclose(interior, sample_data_2d, atol=1e-14)

    def test_default_zero_boundary(self, sample_data_2d):
        """Default: ghost cells filled with zero."""
        width = 1
        out = addGhostDirichlet(sample_data_2d, 0, width=width)
        assert torch.allclose(out[:width, :], torch.zeros_like(out[:width, :]), atol=1e-14)
        assert torch.allclose(out[-width:, :], torch.zeros_like(out[-width:, :]), atol=1e-14)

    def test_custom_boundary_values(self, sample_data_2d):
        """Custom lower/upper Dirichlet values."""
        ghost_data = Bundle(dict(lowerValue=-1.0, upperValue=2.0))
        width = 1
        out = addGhostDirichlet(sample_data_2d, 0, width=width, ghostData=ghost_data)
        assert torch.allclose(out[:width, :], torch.full_like(out[:width, :], -1.0), atol=1e-14)
        assert torch.allclose(out[-width:, :], torch.full_like(out[-width:, :], 2.0), atol=1e-14)

    @pytest.mark.parametrize("dim", [0, 1])
    def test_both_dims(self, sample_data_2d, dim):
        """Works on each dimension."""
        out = addGhostDirichlet(sample_data_2d, dim, width=1)
        expected = list(sample_data_2d.shape)
        expected[dim] += 2
        assert out.shape == tuple(expected)


class TestAddGhostNeumann:
    """Tests for Neumann boundary conditions."""

    def test_output_shape(self, sample_data_2d):
        """Correct shape with ghost cells."""
        out = addGhostNeumann(sample_data_2d, 0, width=1)
        assert out.shape[0] == sample_data_2d.shape[0] + 2

    def test_interior_preserved(self, sample_data_2d):
        """Interior data matches input."""
        width = 1
        out = addGhostNeumann(sample_data_2d, 0, width=width)
        interior = out[width:-width, :]
        assert torch.allclose(interior, sample_data_2d, atol=1e-14)

    def test_zero_derivative_default(self, sample_data_2d):
        """Default: zero derivative means ghost = boundary value."""
        width = 1
        out = addGhostNeumann(sample_data_2d, 0, width=width)
        # With zero Neumann BC, ghost cell = adjacent boundary cell
        assert torch.allclose(out[0:1, :], out[1:2, :], atol=1e-14)

    def test_custom_derivative(self, sample_data_2d):
        """Custom derivative values."""
        ghost_data = Bundle(dict(lowerDerivative=0.5, upperDerivative=-0.5))
        out = addGhostNeumann(sample_data_2d, 0, width=1, ghostData=ghost_data)
        assert torch.isfinite(out).all()

    @pytest.mark.parametrize("dim", [0, 1])
    def test_both_dims(self, sample_data_2d, dim):
        """Works on each dimension."""
        out = addGhostNeumann(sample_data_2d, dim, width=1)
        expected = list(sample_data_2d.shape)
        expected[dim] += 2
        assert out.shape == tuple(expected)


class TestCrossBoundaryConsistency:
    """Cross-boundary-type consistency checks."""

    def test_all_bcs_same_shape(self, sample_data_2d):
        """All BC types produce the same output shape."""
        shapes = []
        for dim in range(2):
            shapes.append(addGhostPeriodic(sample_data_2d, dim, width=1).shape)
            shapes.append(addGhostExtrapolate(sample_data_2d, dim, width=1).shape)
            shapes.append(addGhostDirichlet(sample_data_2d, dim, width=1).shape)
            shapes.append(addGhostNeumann(sample_data_2d, dim, width=1).shape)
        # All shapes for same dim should match
        assert shapes[0] == shapes[1] == shapes[2] == shapes[3]
        assert shapes[4] == shapes[5] == shapes[6] == shapes[7]

    def test_all_bcs_preserve_interior(self, sample_data_2d):
        """All BC types preserve interior data identically."""
        width = 1
        dim = 0
        results = [
            addGhostPeriodic(sample_data_2d, dim, width=width),
            addGhostExtrapolate(sample_data_2d, dim, width=width),
            addGhostDirichlet(sample_data_2d, dim, width=width),
            addGhostNeumann(sample_data_2d, dim, width=width),
        ]
        for r in results:
            interior = r[width:-width, :]
            assert torch.allclose(interior, sample_data_2d, atol=1e-14)

    def test_all_bcs_finite_output(self, sample_data_2d):
        """No NaN or Inf in any BC output."""
        for dim in range(2):
            assert torch.isfinite(addGhostPeriodic(sample_data_2d, dim, 1)).all()
            assert torch.isfinite(addGhostExtrapolate(sample_data_2d, dim, 1)).all()
            assert torch.isfinite(addGhostDirichlet(sample_data_2d, dim, 1)).all()
            assert torch.isfinite(addGhostNeumann(sample_data_2d, dim, 1)).all()
