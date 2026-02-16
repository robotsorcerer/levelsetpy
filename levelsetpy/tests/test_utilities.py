"""
Tests for utility functions and device management.
"""

import pytest
import torch
import numpy as np

from levelsetpy.utilities import (
    Bundle, cell, size, length, numel, ndims, numDims,
    isfield, isbundle, iscell, isnumeric, isfloat, isscalar,
    strcmp, error, ones, zeros, expand, isColumnLength,
    index_array, index_array_torch,
    USE_CUDA, DEVICE, DTYPE,
)


class TestBundle:
    """Tests for the Bundle (matlab-struct-like) class."""

    def test_create_bundle(self):
        b = Bundle(dict(x=1, y=2))
        assert b.x == 1
        assert b.y == 2

    def test_len(self):
        b = Bundle(dict(a=1, b=2, c=3))
        assert len(b) == 3

    def test_setattr(self):
        b = Bundle(dict(x=1))
        b.y = 2
        assert b.y == 2


class TestDeviceManagement:
    """Tests for global device/dtype constants."""

    def test_use_cuda_is_bool(self):
        assert isinstance(USE_CUDA, bool)

    def test_device_is_torch_device(self):
        assert isinstance(DEVICE, torch.device)

    def test_dtype_is_float64(self):
        assert DTYPE == torch.float64

    def test_device_matches_cuda_flag(self):
        if USE_CUDA:
            assert DEVICE.type == 'cuda'
        else:
            assert DEVICE.type == 'cpu'


class TestCellFunction:
    """Tests for the cell() helper."""

    def test_basic_cell(self):
        c = cell(3)
        assert isinstance(c, list)
        assert len(c) == 3

    def test_cell_elements_are_nan(self):
        c = cell(2)
        for elem in c:
            assert np.isnan(elem)


class TestSizeFunction:
    """Tests for the size() helper."""

    def test_size_numpy(self):
        a = np.zeros((3, 4))
        assert size(a) == (3, 4)
        assert size(a, 0) == 3
        assert size(a, 1) == 4

    def test_size_torch(self):
        a = torch.zeros(5, 6)
        assert size(a) == (5, 6)
        assert size(a, 0) == 5


class TestPredicates:
    """Tests for type-checking predicates."""

    def test_isbundle(self):
        assert isbundle(Bundle(dict(x=1)))
        assert not isbundle(dict(x=1))

    def test_isfield(self):
        b = Bundle(dict(x=1, y=2))
        assert isfield(b, 'x')
        assert not isfield(b, 'z')

    def test_iscell(self):
        assert iscell([1, 2, 3])
        assert not iscell((1, 2, 3))

    def test_isnumeric(self):
        assert isnumeric(5)
        assert isnumeric(3.14)
        assert not isnumeric("hello")

    def test_isfloat(self):
        assert isfloat(3.14)
        assert isfloat(np.float64(1.0))
        assert not isfloat(5)

    def test_isscalar(self):
        assert isscalar(5)
        assert isscalar(np.array([1.0]))
        assert not isscalar(np.array([1.0, 2.0]))

    def test_strcmp(self):
        assert strcmp("hello", "hello")
        assert not strcmp("hello", "world")


class TestIndexArrays:
    """Tests for index array generation."""

    def test_index_array_numpy(self):
        idx = index_array(1, 5)
        assert isinstance(idx, np.ndarray)
        np.testing.assert_array_equal(idx, np.array([0, 1, 2, 3, 4]))

    def test_index_array_torch(self):
        idx = index_array_torch(1, 5)
        assert isinstance(idx, torch.Tensor)
        assert idx.dtype == torch.int64
        assert torch.equal(idx, torch.arange(0, 5, dtype=torch.int64))


class TestErrorFunction:
    """Tests for the error() helper."""

    def test_raises_value_error(self):
        with pytest.raises(ValueError, match="test error"):
            error("test error")


class TestGridCreation:
    """Tests for grid creation and consistency."""

    def test_2d_grid_fields(self, grid_2d):
        """Grid has all required fields."""
        assert hasattr(grid_2d, 'dim')
        assert hasattr(grid_2d, 'min')
        assert hasattr(grid_2d, 'max')
        assert hasattr(grid_2d, 'N')
        assert hasattr(grid_2d, 'dx')
        assert hasattr(grid_2d, 'vs')
        assert hasattr(grid_2d, 'xs')
        assert hasattr(grid_2d, 'bdry')
        assert hasattr(grid_2d, 'shape')

    def test_2d_grid_dimension(self, grid_2d):
        assert grid_2d.dim == 2

    def test_3d_grid_dimension(self, grid_3d):
        assert grid_3d.dim == 3

    def test_xs_are_torch_tensors(self, grid_2d):
        for xs in grid_2d.xs:
            assert isinstance(xs, torch.Tensor)

    def test_grid_shape_consistency(self, grid_2d):
        """grid.shape matches grid.xs shapes."""
        for i, xs in enumerate(grid_2d.xs):
            assert xs.shape == grid_2d.shape, f"xs[{i}] shape mismatch"

    def test_dx_positive(self, grid_2d):
        """Grid spacing is positive."""
        assert (grid_2d.dx > 0).all()

    def test_1d_grid(self, grid_1d):
        assert grid_1d.dim == 1
        assert len(grid_1d.xs) == 1
