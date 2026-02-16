"""
Shared pytest fixtures for the LevelSetPy test suite.
"""

import os

# Exclude legacy test files that have module-level code causing import errors
collect_ignore = [
    os.path.join(os.path.dirname(__file__), "test_grids.py"),
    os.path.join(os.path.dirname(__file__), "test_mesh_2d.py"),
    os.path.join(os.path.dirname(__file__), "test_mesh_3d.py"),
    os.path.join(os.path.dirname(__file__), "test_mesh_2d_only.py"),
]

import pytest
import torch
import numpy as np
from math import pi

from levelsetpy.grids import createGrid
from levelsetpy.utilities import Bundle
from levelsetpy.boundarycondition import addGhostPeriodic, addGhostExtrapolate
from levelsetpy.boundarycondition.add_ghost_neumann import addGhostNeumann
from levelsetpy.boundarycondition.add_ghost_dirichlet import addGhostDirichlet


@pytest.fixture
def grid_2d():
    """A standard 2D grid on [-1, 1]^2 with 51 nodes per dimension."""
    gmin = -1.0 * np.ones((2, 1), dtype=np.float64)
    gmax = +1.0 * np.ones((2, 1), dtype=np.float64)
    N = 51 * np.ones((2, 1), dtype=np.int64)
    g = createGrid(gmin, gmax, N, process=True)
    g.xs = [torch.as_tensor(x) for x in g.xs]
    return g


@pytest.fixture
def grid_2d_periodic():
    """A 2D grid with periodic boundary conditions on both dimensions."""
    gmin = -1.0 * np.ones((2, 1), dtype=np.float64)
    gmax = +1.0 * np.ones((2, 1), dtype=np.float64)
    N = 51 * np.ones((2, 1), dtype=np.int64)
    g = createGrid(gmin, gmax, N, process=True)
    g.bdry = [addGhostPeriodic, addGhostPeriodic]
    g.bdryData = [None, None]
    g.xs = [torch.as_tensor(x) for x in g.xs]
    return g


@pytest.fixture
def grid_3d():
    """A standard 3D grid for Dubins-style problems."""
    gmin = np.array([[-1.0, -1.0, -pi]]).T
    gmax = np.array([[1.0, 1.0, pi]]).T
    N = 21 * np.ones((3, 1), dtype=np.int64)
    pdDims = 2  # 3rd dimension is periodic
    g = createGrid(gmin, gmax, N, pdDims)
    g.xs = [torch.as_tensor(x) for x in g.xs]
    return g


@pytest.fixture
def grid_1d():
    """A 1D grid on [-1, 1] with 101 nodes."""
    gmin = np.array([[-1.0]]).T
    gmax = np.array([[1.0]]).T
    N = 101 * np.ones((1, 1), dtype=np.int64)
    g = createGrid(gmin, gmax, N, process=True)
    g.xs = [torch.as_tensor(x) for x in g.xs]
    return g


@pytest.fixture
def sample_data_2d(grid_2d):
    """A smooth 2D test function: sin(pi*x)*cos(pi*y)."""
    data = torch.sin(pi * grid_2d.xs[0]) * torch.cos(pi * grid_2d.xs[1])
    return data.to(torch.float64)


@pytest.fixture
def linear_data_2d(grid_2d):
    """A linear 2D test function: 2*x + 3*y. Exact derivatives are known."""
    data = 2.0 * grid_2d.xs[0] + 3.0 * grid_2d.xs[1]
    return data.to(torch.float64)


@pytest.fixture
def sphere_data_2d(grid_2d):
    """Signed distance function for a unit circle centered at origin."""
    data = torch.sqrt(grid_2d.xs[0]**2 + grid_2d.xs[1]**2) - 0.5
    return data.to(torch.float64)


@pytest.fixture
def grid_2d_wide():
    """A 2D grid on [-2, 2]^2 with 51 nodes (wider domain for shape sign changes)."""
    gmin = -2.0 * np.ones((2, 1), dtype=np.float64)
    gmax = +2.0 * np.ones((2, 1), dtype=np.float64)
    N = 51 * np.ones((2, 1), dtype=np.int64)
    g = createGrid(gmin, gmax, N, process=True)
    g.xs = [torch.as_tensor(x) for x in g.xs]
    return g


@pytest.fixture
def grid_3d_wide():
    """A 3D grid on [-2, 2]^3 with 21 nodes."""
    gmin = -2.0 * np.ones((3, 1), dtype=np.float64)
    gmax = +2.0 * np.ones((3, 1), dtype=np.float64)
    N = 21 * np.ones((3, 1), dtype=np.int64)
    g = createGrid(gmin, gmax, N, process=True)
    g.xs = [torch.as_tensor(x) for x in g.xs]
    return g


@pytest.fixture
def grid_1d_wide():
    """A 1D grid on [-2, 2] with 101 nodes."""
    gmin = np.array([[-2.0]]).T
    gmax = np.array([[2.0]]).T
    N = 101 * np.ones((1, 1), dtype=np.int64)
    g = createGrid(gmin, gmax, N, process=True)
    g.xs = [torch.as_tensor(x) for x in g.xs]
    return g
