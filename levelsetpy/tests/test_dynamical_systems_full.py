"""
Tests for dynamical system classes beyond DubinsVehicleRel.
Tests: DoubleIntegrator, DubinsVehicleAbs, Bird, RocketSystemRel.
"""

import pytest
import torch
import numpy as np
from math import pi

from levelsetpy.grids import createGrid
from levelsetpy.utilities import Bundle, DTYPE, deg2rad
from levelsetpy.dynamicalsystems import (
    DoubleIntegrator,
    DubinsVehicleAbs,
    Bird,
    RocketSystemRel,
)


@pytest.fixture
def dint_grid():
    """A 2D grid for the double integrator (position x velocity)."""
    gmin = -2.0 * np.ones((2, 1), dtype=np.float64)
    gmax = +2.0 * np.ones((2, 1), dtype=np.float64)
    N = 31 * np.ones((2, 1), dtype=np.int64)
    g = createGrid(gmin, gmax, N, process=True)
    g.xs = [torch.as_tensor(x, dtype=DTYPE) for x in g.xs]
    return g


@pytest.fixture
def rocket_grid():
    """A 3D grid for rocket pursuit-evasion."""
    gmin = np.array([[-2.0, -2.0, -pi]]).T
    gmax = np.array([[2.0, 2.0, pi]]).T
    N = 15 * np.ones((3, 1), dtype=np.int64)
    gmax[2, 0] *= (1 - 2 / N[2, 0])
    pdDims = 2
    g = createGrid(gmin, gmax, N, pdDims)
    g.xs = [torch.as_tensor(x, dtype=DTYPE) for x in g.xs]
    return g


@pytest.fixture
def bird_grid():
    """A 3D grid for Bird agents."""
    gmin = np.array([[-1.0, -1.0, -pi]]).T
    gmax = np.array([[1.0, 1.0, pi]]).T
    N = 15 * np.ones((3, 1), dtype=np.int64)
    gmax[2, 0] *= (1 - 2 / N[2, 0])
    pdDims = 2
    g = createGrid(gmin, gmax, N, pdDims)
    g.xs = [torch.as_tensor(x, dtype=DTYPE) for x in g.xs]
    return g


class TestDoubleIntegrator:
    """Tests for the DoubleIntegrator dynamical system."""

    def test_initialization(self, dint_grid):
        """Constructor works and Gamma is computed."""
        di = DoubleIntegrator(dint_grid, u_bound=1)
        assert hasattr(di, 'Gamma')
        assert hasattr(di, 'control_law')
        assert di.control_law == 1

    def test_switching_curve_shape(self, dint_grid):
        """Gamma.shape matches grid.shape."""
        di = DoubleIntegrator(dint_grid, u_bound=1)
        gamma = di.Gamma
        if isinstance(gamma, torch.Tensor):
            assert gamma.shape == dint_grid.shape
        else:
            assert np.asarray(gamma).shape == dint_grid.shape

    def test_switching_curve_values(self, dint_grid):
        """Gamma = -0.5 * x2 * |x2| at known points."""
        di = DoubleIntegrator(dint_grid, u_bound=1)
        gamma = di.Gamma
        xs_np = [x.numpy() if isinstance(x, torch.Tensor) else np.asarray(x)
                 for x in dint_grid.xs]
        expected = -0.5 * xs_np[1] * np.abs(xs_np[1])
        gamma_np = gamma.numpy() if isinstance(gamma, torch.Tensor) else np.asarray(gamma)
        np.testing.assert_allclose(gamma_np, expected, atol=1e-10)

    def test_hamiltonian_shape_and_finite(self, dint_grid):
        """Hamiltonian has grid shape and all values are finite."""
        di = DoubleIntegrator(dint_grid, u_bound=1)
        derivs = [torch.ones(dint_grid.shape, dtype=DTYPE) for _ in range(2)]
        data = torch.zeros(dint_grid.shape, dtype=DTYPE)
        H = di.hamiltonian(0.0, data, derivs, None)
        assert H.shape == dint_grid.shape
        assert torch.isfinite(H).all()

    def test_dissipation_dim0(self, dint_grid):
        """dim=0 dissipation returns |x2|."""
        di = DoubleIntegrator(dint_grid, u_bound=1)
        alpha = di.dissipation(0, None, None, None, None, 0)
        expected = torch.abs(torch.as_tensor(dint_grid.xs[1]))
        assert torch.allclose(alpha, expected, atol=1e-10)

    def test_dissipation_dim1(self, dint_grid):
        """dim=1 dissipation returns |u_bound|."""
        di = DoubleIntegrator(dint_grid, u_bound=2)
        alpha = di.dissipation(0, None, None, None, None, 1)
        assert torch.isfinite(alpha).all() if isinstance(alpha, torch.Tensor) else True
        # Should be abs(u_bound) = 2
        expected_val = 2.0
        if isinstance(alpha, torch.Tensor):
            assert alpha.item() == pytest.approx(expected_val)
        else:
            assert float(alpha) == pytest.approx(expected_val)


class TestDubinsVehicleAbs:
    """Tests for DubinsVehicleAbs (has known init_random bug)."""

    def test_initialization(self, bird_grid):
        """Constructor works with default rw_cov=0."""
        dv = DubinsVehicleAbs(bird_grid, u_bound=1, w_bound=1,
                              init_state=[0, 0, 0], label="test")
        assert hasattr(dv, 'state')
        assert hasattr(dv, 'grid')

    def test_methods_exist(self):
        """Verify expected methods exist on the class."""
        assert hasattr(DubinsVehicleAbs, 'dissipation')
        assert hasattr(DubinsVehicleAbs, 'dynamics')
        assert hasattr(DubinsVehicleAbs, 'initialize')
        assert hasattr(DubinsVehicleAbs, 'update_values')


class TestBird:
    """Tests for the Bird (single agent in a flock) class."""

    @pytest.fixture
    def bird_pair(self, bird_grid):
        """Two Bird instances for neighbor tests."""
        init0 = np.array([[0.5, 0.3, 0.1]])
        init1 = np.array([[0.2, -0.1, 0.5]])
        b0 = Bird(bird_grid, u_bound=1, w_bound=deg2rad(10),
                  init_xyw=init0, label=0)
        b1 = Bird(bird_grid, u_bound=1, w_bound=deg2rad(10),
                  init_xyw=init1, label=1)
        return b0, b1

    def test_initialization(self, bird_grid):
        """Bird constructs with valid init_xyw."""
        init_xyw = np.array([[0.5, 0.3, 0.1]])
        b = Bird(bird_grid, u_bound=1, w_bound=deg2rad(10),
                 init_xyw=init_xyw, label=0)
        assert hasattr(b, 'cur_state')
        assert hasattr(b, 'label')
        assert b.label == 0

    def test_neighbor_management(self, bird_pair):
        """update_neighbor and has_neighbor work correctly."""
        b0, b1 = bird_pair
        assert not b0.has_neighbor(), "Should start with no neighbors"
        b0.update_neighbor(b1)
        assert b0.has_neighbor(), "Should have a neighbor after update"
        assert b0.valence == 1

    def test_no_self_neighbor(self, bird_pair):
        """Can't add self as neighbor."""
        b0, _ = bird_pair
        b0.update_neighbor(b0)
        assert b0.valence == 0, "Should not add self as neighbor"

    def test_hash_equality(self, bird_grid):
        """Same label -> equal, different label -> not equal."""
        init = np.array([[0.5, 0.3, 0.1]])
        b0 = Bird(bird_grid, u_bound=1, w_bound=deg2rad(10),
                  init_xyw=init, label=42)
        b1 = Bird(bird_grid, u_bound=1, w_bound=deg2rad(10),
                  init_xyw=init, label=42)
        b2 = Bird(bird_grid, u_bound=1, w_bound=deg2rad(10),
                  init_xyw=init, label=99)
        assert b0 == b1, "Same label should be equal"
        assert b0 != b2, "Different labels should not be equal"

    def test_dynamics_abs_shape(self, bird_pair):
        """dynamics_abs returns array of correct shape."""
        b0, _ = bird_pair
        cur_state = b0.cur_state
        result = b0.dynamics_abs(cur_state)
        assert isinstance(result, np.ndarray)
        assert result.shape == cur_state.shape


class TestRocketSystemRel:
    """Tests for the RocketSystemRel dynamical system."""

    def test_initialization(self, rocket_grid):
        """Constructor works with grid and params."""
        rs = RocketSystemRel(rocket_grid, u_bound=5, w_bound=5)
        assert hasattr(rs, 'grid')
        assert hasattr(rs, 'u_e')
        assert hasattr(rs, 'u_p')

    def test_hamiltonian_shape_and_finite(self, rocket_grid):
        """Hamiltonian has correct shape and all finite."""
        rs = RocketSystemRel(rocket_grid, u_bound=5, w_bound=5)
        derivs = [torch.ones(rocket_grid.shape, dtype=DTYPE) for _ in range(3)]
        data = torch.zeros(rocket_grid.shape, dtype=DTYPE)
        H = rs.hamiltonian(0.0, data, derivs, None)
        assert H.shape == rocket_grid.shape
        assert torch.isfinite(H).all()

    def test_dissipation_dim0_dim1(self, rocket_grid):
        """Dissipation works for dim 0 and dim 1."""
        rs = RocketSystemRel(rocket_grid, u_bound=5, w_bound=5)
        for dim in range(2):
            alpha = rs.dissipation(0, None, None, None, None, dim)
            assert torch.isfinite(alpha).all(), f"Dissipation NaN at dim={dim}"

    def test_dissipation_dim2(self, rocket_grid):
        """Dissipation for dim=2 (scalar result)."""
        rs = RocketSystemRel(rocket_grid, u_bound=5, w_bound=5)
        alpha = rs.dissipation(0, None, None, None, None, 2)
        assert isinstance(alpha, (int, float)), "dim=2 dissipation should be scalar"

    def test_dynamics_returns_list(self, rocket_grid):
        """dynamics() returns 3-element list."""
        rs = RocketSystemRel(rocket_grid, u_bound=5, w_bound=5)
        xdot = rs.dynamics()
        assert isinstance(xdot, list)
        assert len(xdot) == 3
