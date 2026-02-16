"""
Tests for dynamical system classes.
Tests: DubinsVehicleRel and its hamiltonian/dissipation/dynamics methods.
"""

import pytest
import torch
import numpy as np
from math import pi

from levelsetpy.grids import createGrid
from levelsetpy.utilities import Bundle, DTYPE
from levelsetpy.dynamicalsystems import DubinsVehicleRel


@pytest.fixture
def dubins_grid():
    """A 3D grid for Dubins vehicle problems."""
    gmin = np.array([[-1.0, -1.0, -pi]]).T
    gmax = np.array([[1.0, 1.0, pi]]).T
    N = 15 * np.ones((3, 1), dtype=np.int64)
    gmax[2, 0] *= (1 - 2 / N[2, 0])
    pdDims = 2
    g = createGrid(gmin, gmax, N, pdDims)
    g.xs = [torch.as_tensor(x, dtype=DTYPE) for x in g.xs]
    return g


class TestDubinsVehicleRel:
    """Tests for the Dubins vehicle in relative coordinates."""

    def test_initialization(self, dubins_grid):
        """DubinsVehicleRel initializes without error."""
        dv = DubinsVehicleRel(dubins_grid, u_bound=1, w_bound=1)
        assert hasattr(dv, 'v_e')
        assert hasattr(dv, 'v_p')
        assert hasattr(dv, 'w_e')
        assert hasattr(dv, 'w_p')

    def test_hamiltonian_output_shape(self, dubins_grid):
        """Hamiltonian returns array matching grid shape."""
        dv = DubinsVehicleRel(dubins_grid, u_bound=1, w_bound=1)
        # Create mock derivatives
        derivC = [torch.ones(dubins_grid.shape, dtype=DTYPE) for _ in range(3)]
        data = torch.zeros(dubins_grid.shape, dtype=DTYPE)

        H = dv.hamiltonian(0.0, data, derivC, None)
        assert H.shape == dubins_grid.shape
        assert torch.isfinite(H).all()

    def test_hamiltonian_value(self, dubins_grid):
        """Hamiltonian produces finite, reasonable values."""
        dv = DubinsVehicleRel(dubins_grid, u_bound=1, w_bound=1)
        derivC = [0.5 * torch.ones(dubins_grid.shape, dtype=DTYPE) for _ in range(3)]
        data = torch.zeros(dubins_grid.shape, dtype=DTYPE)

        H = dv.hamiltonian(0.0, data, derivC, None)
        assert torch.isfinite(H).all()
        assert H.abs().max() < 1e6

    def test_dissipation_dim0(self, dubins_grid):
        """Dissipation for dim=0 is finite and non-negative."""
        dv = DubinsVehicleRel(dubins_grid, u_bound=1, w_bound=1)
        alpha = dv.dissipation(0, None, None, None, None, 0)
        assert isinstance(alpha, torch.Tensor)
        assert torch.isfinite(alpha).all()

    def test_dissipation_dim1(self, dubins_grid):
        """Dissipation for dim=1."""
        dv = DubinsVehicleRel(dubins_grid, u_bound=1, w_bound=1)
        alpha = dv.dissipation(0, None, None, None, None, 1)
        assert torch.isfinite(alpha).all()

    def test_dissipation_dim2(self, dubins_grid):
        """Dissipation for dim=2 is a scalar."""
        dv = DubinsVehicleRel(dubins_grid, u_bound=1, w_bound=1)
        alpha = dv.dissipation(0, None, None, None, None, 2)
        # dim=2 returns w_e + w_p which is a scalar
        assert alpha > 0

    def test_dissipation_invalid_dim(self, dubins_grid):
        """Invalid dim raises assertion."""
        dv = DubinsVehicleRel(dubins_grid, u_bound=1, w_bound=1)
        with pytest.raises(AssertionError):
            dv.dissipation(0, None, None, None, None, 3)

    def test_dynamics_output(self, dubins_grid):
        """dynamics() returns 3-element list."""
        dv = DubinsVehicleRel(dubins_grid, u_bound=1, w_bound=1)
        xdot = dv.dynamics()
        assert isinstance(xdot, list)
        assert len(xdot) == 3
        # First two are tensors (grid-dependent), third may be scalar constant
        for i in range(2):
            assert isinstance(xdot[i], torch.Tensor), f"xdot[{i}] not a tensor"
            assert torch.isfinite(xdot[i]).all(), f"xdot[{i}] has NaN/Inf"

    def test_dynamics_uses_torch_trig(self, dubins_grid):
        """dynamics() should use torch.cos/sin, not np.cos/sin."""
        dv = DubinsVehicleRel(dubins_grid, u_bound=1, w_bound=1)
        xdot = dv.dynamics()
        # First two elements use trig and should be torch.Tensor
        assert isinstance(xdot[0], torch.Tensor)
        assert isinstance(xdot[1], torch.Tensor)

    def test_different_speed_bounds(self, dubins_grid):
        """Different u_bound and w_bound values work."""
        dv = DubinsVehicleRel(dubins_grid, u_bound=5, w_bound=3)
        H = dv.hamiltonian(
            0.0,
            torch.zeros(dubins_grid.shape, dtype=DTYPE),
            [torch.ones(dubins_grid.shape, dtype=DTYPE) for _ in range(3)],
            None,
        )
        assert torch.isfinite(H).all()
