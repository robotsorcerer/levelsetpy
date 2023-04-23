__all__ = [
    "cells_from_grid"
]

__author__ 		= "Lekan Molu"
__copyright__ 	= "2021, Hamilton-Jacobi Analysis in Python"
__credits__  	= "There are None."
__license__ 	= "Molux Licence"
__maintainer__ 	= "Lekan Molu"
__email__ 		= "patlekno@icloud.com"
__status__ 		= "Completed"


import copy
import numpy as np
from .cell_neighs import neighbors
from LevelSetPy.Utilities import *
from LevelSetPy.Grids import getOGPBounds, createGrid


def cells_from_grid(g, bounds, padding=None):
    """
     gs = cells_from_grid(g, bounds, padding)
         Splits the grid into smaller grids, each with specified bounds.
         Optionally, padding can be specified so that the grids overlap

     Inp.ts:
         g      - original grid
         bounds - list of bounds of the smaller grids. This should be a g.dim
                  dimensional matrix that specifies the "grid" of bounds.
             For example, suppose the original grid is a [-1, 1]^2 grid in 2D.
             Then, the following bounds would split it into [-1, 0]^2, [0, 1]^2,
             [-1, 0] x [0, 1], and [0, 1] x [-1, 0] grids:
                 bounds = {[-1, 0, 1], [-1, 0, 1]};
         padding - amount of overlap between two adjacent subgrids

     Output:
         gs - subgrids

     Author: Lekan Molu, September 04, 2021
     """
    if padding is None:
        padding = np.zeros((g.dim, 1))

    assert isinstance(bounds, list), 'bounds must be a list or list of lists'
    ## Create a grid for the bounds
    if g.dim > 1:
        bounds_grid = np.meshgrid(*bounds, sparse=False, indexing='ij');
    else:
        # indexing and sparse flags have no effect in 1D case
        bounds_grid = np.meshgrid(bounds, indexing='ij')[0]

    ## Create grids based on the bound grid
    temp = size(bounds_grid[0])
    temparr = np.array((temp))
    gs = np.zeros(temparr-(temparr>1).astype(np.int64))

    ii = cell(g.dim, 1)
    gss = []
    partition = {}
    for i in range(numel(gs)):
        ii = np.asarray(np.unravel_index(i, size(gs), order='F'))
        iip = copy.copy(ii)
        # print('iip: ', iip)
        for j in range(g.dim):
            iip[j] += 1
        grid_min = []
        grid_max = []
        # turn'em to indices (tuples) to aid dynamic
        # indexing (see: https://numpy.org/doc/stable/user/basics.indexing.html)
        ii, iip = tuple(ii), tuple(iip)
        for j in range(g.dim):
            grid_min.append(bounds_grid[j][ii])
            grid_max.append(bounds_grid[j][iip])
        grid_min, grid_max = np.vstack((grid_min)), np.vstack((grid_max))
        grid_min, grid_max, N = getOGPBounds(g, grid_min, grid_max, padding)

        # create cell within grid
        celi = createGrid(grid_min, grid_max, N, process=True)
        celi.neighs = neighbors(ii, gs.shape) # neighbors of this cell
        celi.idx = ii # index of this cell within the grid subgrd
        celi.gshape = gs.shape # shape of containing grid
        gss.append(celi)
        
    return gss
