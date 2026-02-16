"""
Tests for the initial conditions (shape) functions.
Tests: shapeSphere, shapeCylinder, shapeEllipsoid, shapeRectangleByCenter,
       shapeRectangleByCorners, shapeUnion, shapeIntersection,
       shapeComplement, shapeDifference.
"""

import copy
import pytest
import torch
import numpy as np
from math import pi

from levelsetpy.grids import createGrid
from levelsetpy.initialconditions import (
    shapeSphere,
    shapeCylinder,
    shapeEllipsoid,
    shapeRectangleByCenter,
    shapeRectangleByCorners,
    shapeUnion,
    shapeIntersection,
    shapeComplement,
    shapeDifference,
)


def _np_grid(grid):
    """Return a shallow copy of grid with xs converted to numpy arrays.

    Initial condition functions use numpy operations (np.sqrt, np.maximum, np.all)
    that are incompatible with torch tensors in numpy 2.0+.
    """
    g = copy.copy(grid)
    g.xs = [x.numpy() if isinstance(x, torch.Tensor) else np.asarray(x)
            for x in g.xs]
    return g


def _to_numpy(data):
    """Convert data to numpy array regardless of type."""
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    return np.asarray(data)


# ── Fixtures (numpy grids for initial condition functions) ──

@pytest.fixture
def np_grid_2d():
    """2D grid on [-2,2]^2, N=51, with numpy xs."""
    gmin = -2.0 * np.ones((2, 1), dtype=np.float64)
    gmax = +2.0 * np.ones((2, 1), dtype=np.float64)
    N = 51 * np.ones((2, 1), dtype=np.int64)
    return createGrid(gmin, gmax, N, process=True)


@pytest.fixture
def np_grid_3d():
    """3D grid on [-2,2]^3, N=21, with numpy xs."""
    gmin = -2.0 * np.ones((3, 1), dtype=np.float64)
    gmax = +2.0 * np.ones((3, 1), dtype=np.float64)
    N = 21 * np.ones((3, 1), dtype=np.int64)
    return createGrid(gmin, gmax, N, process=True)


@pytest.fixture
def np_grid_1d():
    """1D grid on [-2,2], N=101, with numpy xs."""
    gmin = np.array([[-2.0]]).T
    gmax = np.array([[2.0]]).T
    N = 101 * np.ones((1, 1), dtype=np.int64)
    return createGrid(gmin, gmax, N, process=True)


class TestShapeSphere:
    """Tests for the shapeSphere function."""

    def test_output_shape_2d(self, np_grid_2d):
        """Output shape matches grid.shape."""
        data = shapeSphere(np_grid_2d)
        data_np = _to_numpy(data)
        assert data_np.shape == np_grid_2d.shape

    def test_signed_distance_values(self, np_grid_2d):
        """Origin should be negative (inside), far corner positive (outside)."""
        g = np_grid_2d
        data = _to_numpy(shapeSphere(g, radius=1.0))
        # Find the grid point closest to the origin
        dist_to_origin = np.sqrt(g.xs[0] ** 2 + g.xs[1] ** 2)
        idx_origin = np.unravel_index(np.argmin(dist_to_origin), dist_to_origin.shape)
        assert data[idx_origin] == pytest.approx(-1.0, abs=0.1)
        assert data[0, 0] > 0, "Corner should be outside the sphere"

    def test_sign_change_exists(self, np_grid_2d):
        """Both positive and negative values should exist."""
        data = _to_numpy(shapeSphere(np_grid_2d, radius=1.0))
        assert np.any(data < 0), "No interior points"
        assert np.any(data > 0), "No exterior points"

    def test_custom_center(self, np_grid_2d):
        """Non-origin center shifts the sphere."""
        center = np.array([[0.5], [0.5]])
        data = _to_numpy(shapeSphere(np_grid_2d, center=center, radius=0.5))
        dist_to_center = np.sqrt(
            (np_grid_2d.xs[0] - 0.5) ** 2 + (np_grid_2d.xs[1] - 0.5) ** 2
        )
        idx_center = np.unravel_index(np.argmin(dist_to_center), dist_to_center.shape)
        assert data[idx_center] < 0, "Center should be inside sphere"

    def test_1d_interval(self, np_grid_1d):
        """1D sphere is an interval."""
        data = _to_numpy(shapeSphere(np_grid_1d, radius=1.0)).flatten()
        assert np.any(data < 0), "No interior in 1D"
        assert np.any(data > 0), "No exterior in 1D"

    def test_3d_output_shape(self, np_grid_3d):
        """Works on 3D grid."""
        data = _to_numpy(shapeSphere(np_grid_3d, radius=1.0))
        assert data.shape == np_grid_3d.shape


class TestShapeCylinder:
    """Tests for the shapeCylinder function."""

    def test_3d_axis_aligned(self, np_grid_3d):
        """Cylinder along z-axis has correct shape and signs."""
        data = _to_numpy(shapeCylinder(np_grid_3d, axis_align=2, radius=1.0))
        assert data.shape == np_grid_3d.shape
        assert np.any(data < 0), "No interior"
        assert np.any(data > 0), "No exterior"

    def test_signed_distance_at_origin(self, np_grid_3d):
        """Origin value should be about -radius."""
        g = np_grid_3d
        data = _to_numpy(shapeCylinder(g, axis_align=2, radius=1.0))
        dist = np.sqrt(g.xs[0] ** 2 + g.xs[1] ** 2 + g.xs[2] ** 2)
        idx_origin = np.unravel_index(np.argmin(dist), dist.shape)
        assert data[idx_origin] == pytest.approx(-1.0, abs=0.25)

    def test_empty_align_equals_sphere(self, np_grid_3d):
        """axis_align=[] makes cylinder degenerate to sphere."""
        data_cyl = _to_numpy(shapeCylinder(np_grid_3d, axis_align=[], radius=1.0))
        data_sph = _to_numpy(shapeSphere(np_grid_3d, radius=1.0))
        np.testing.assert_allclose(data_cyl, data_sph, atol=1e-12)

    def test_2d_matches_sphere(self, np_grid_2d):
        """On 2D grid with no axis_align dimensions, equivalent to sphere."""
        data_cyl = _to_numpy(shapeCylinder(np_grid_2d, axis_align=[], radius=1.0))
        data_sph = _to_numpy(shapeSphere(np_grid_2d, radius=1.0))
        np.testing.assert_allclose(data_cyl, data_sph, atol=1e-12)


class TestShapeEllipsoid:
    """Tests for the shapeEllipsoid function."""

    def test_output_shape(self, np_grid_2d):
        """Output shape matches grid."""
        data = _to_numpy(shapeEllipsoid(np_grid_2d, radius=1.0))
        assert data.shape == np_grid_2d.shape

    def test_sign_change(self, np_grid_2d):
        """Interior and exterior exist."""
        data = _to_numpy(shapeEllipsoid(np_grid_2d, radius=1.0))
        assert np.any(data < 0), "No interior"
        assert np.any(data > 0), "No exterior"

    def test_anisotropy_xy(self, np_grid_2d):
        """Ellipsoid has x^2 + 4*y^2 weighting, so y-axis is compressed."""
        data = _to_numpy(shapeEllipsoid(np_grid_2d, radius=1.0))
        # Slice at x ≈ 0 (middle row)
        y_slice = data[data.shape[0] // 2, :]
        sign_changes = np.where(np.diff(np.sign(y_slice)))[0]
        assert len(sign_changes) >= 1, "No sign change found along y-axis"

    def test_3d_weights(self, np_grid_3d):
        """3D ellipsoid includes 9*z^2 term."""
        data = _to_numpy(shapeEllipsoid(np_grid_3d, radius=1.0))
        assert data.shape == np_grid_3d.shape
        assert np.any(data < 0) and np.any(data > 0)


class TestShapeRectangle:
    """Tests for rectangle shape functions."""

    def test_rect_by_corners_2d(self, np_grid_2d):
        """Rectangle by corners: origin negative, outside positive."""
        g = np_grid_2d
        lower = np.array([[-0.5], [-0.5]])
        upper = np.array([[0.5], [0.5]])
        data = _to_numpy(shapeRectangleByCorners(g, lower, upper))
        assert data.shape == g.shape
        dist = np.sqrt(g.xs[0] ** 2 + g.xs[1] ** 2)
        idx_origin = np.unravel_index(np.argmin(dist), dist.shape)
        assert data[idx_origin] < 0, "Origin should be inside rectangle"
        assert data[0, 0] > 0, "Far corner should be outside rectangle"

    def test_rect_by_center_2d(self, np_grid_2d):
        """Rectangle by center: works with center/width params."""
        center = np.array([[0.0], [0.0]])
        widths = np.array([[1.0], [1.0]])
        data = _to_numpy(shapeRectangleByCenter(np_grid_2d, center, widths))
        assert data.shape == np_grid_2d.shape
        assert np.any(data < 0) and np.any(data > 0)

    def test_center_matches_corners(self, np_grid_2d):
        """Center/width and corners versions agree for equivalent params."""
        center = np.array([[0.0], [0.0]])
        widths = np.array([[1.0], [1.0]])
        lower = np.array([[-0.5], [-0.5]])
        upper = np.array([[0.5], [0.5]])
        data_center = _to_numpy(shapeRectangleByCenter(np_grid_2d, center, widths))
        data_corners = _to_numpy(shapeRectangleByCorners(np_grid_2d, lower, upper))
        np.testing.assert_allclose(data_center, data_corners, atol=1e-12)

    def test_1d_interval(self, np_grid_1d):
        """1D rectangle is an interval."""
        lower = np.array([[-0.5]])
        upper = np.array([[0.5]])
        data = _to_numpy(shapeRectangleByCorners(np_grid_1d, lower, upper)).flatten()
        assert np.any(data < 0) and np.any(data > 0)


class TestShapeOperations:
    """Tests for shape set operations."""

    @pytest.fixture
    def two_spheres(self, np_grid_2d):
        """Two overlapping spheres for set operation tests."""
        c1 = np.array([[-0.5], [0.0]])
        c2 = np.array([[0.5], [0.0]])
        s1 = _to_numpy(shapeSphere(np_grid_2d, center=c1, radius=0.8))
        s2 = _to_numpy(shapeSphere(np_grid_2d, center=c2, radius=0.8))
        return s1, s2

    def test_union_is_pointwise_min(self, two_spheres):
        """shapeUnion([s1, s2]) == np.minimum(s1, s2)."""
        s1, s2 = two_spheres
        result = shapeUnion([s1, s2])
        expected = np.minimum(s1, s2)
        np.testing.assert_allclose(result, expected, atol=1e-12)

    def test_intersection_is_pointwise_max(self, two_spheres):
        """shapeIntersection([s1, s2]) == np.maximum(s1, s2)."""
        s1, s2 = two_spheres
        result = shapeIntersection([s1, s2])
        expected = np.maximum(s1, s2)
        np.testing.assert_allclose(result, expected, atol=1e-12)

    def test_complement_is_negation(self, two_spheres):
        """shapeComplement(s) == -s."""
        s1, _ = two_spheres
        result = shapeComplement(s1)
        np.testing.assert_allclose(result, -s1, atol=1e-12)

    def test_difference_two_shapes(self, two_spheres):
        """shapeDifference([s1, s2]) == np.maximum(s1, -s2)."""
        s1, s2 = two_spheres
        result = shapeDifference([s1, s2])
        expected = np.maximum(s1, -s2)
        np.testing.assert_allclose(result, expected, atol=1e-12)

    def test_union_interior_contains_parts(self, two_spheres):
        """Union has at least as many interior points as each part."""
        s1, s2 = two_spheres
        union = shapeUnion([s1, s2])
        assert np.sum(union < 0) >= np.sum(s1 < 0)
        assert np.sum(union < 0) >= np.sum(s2 < 0)

    def test_intersection_interior_subset(self, two_spheres):
        """Intersection has at most as many interior points as each part."""
        s1, s2 = two_spheres
        inter = shapeIntersection([s1, s2])
        assert np.sum(inter < 0) <= np.sum(s1 < 0)
        assert np.sum(inter < 0) <= np.sum(s2 < 0)
