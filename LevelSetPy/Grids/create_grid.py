__all__ = ["createGrid"]

__author__ 		= "Lekan Molu"
__copyright__ 	= "2021, Hamilton-Jacobi Analysis in Python"
__credits__  	= "There are None."
__license__ 	= "Molux Licence"
__maintainer__ 	= "Lekan Molu"
__email__ 		= "patlekno@icloud.com"
__status__ 		= "Completed"


import logging
import numpy as np
from LevelSetPy.Utilities import *
from .process_grid import processGrid
from LevelSetPy.BoundaryCondition import addGhostExtrapolate, addGhostPeriodic

logger = logging.getLogger(__name__)


def  createGrid(grid_min, grid_max, N, pdDims=None, process=True, low_mem=False):
    """
     g = createGrid(grid_min, grid_max, N, pdDim)

     Thin wrapper around processGrid to create a grid compatible with the
     level set toolbox

     Inp.ts:
       grid_min, grid_max - minimum and maximum bounds on computation domain
       N                  - number of grid points in each dimension
       pdDims             - periodic dimensions (eg. pdDims = [2 3] if 2nd and
                                                    3rd dimensions are periodic)
       process            - specifies whether to call processGrid to generate
                            grid points

     Output:
       g - grid structure
    """

    if not pdDims:
        pdDims = []

    # Inp.t checks
    if isscalar(N):
        N = N*np.ones(grid_min.shape).astype(np.int64)

    if not isvector(grid_min) or not isvector(grid_max) or not isvector(N):
        logger.fatal('grid_min, grid_max, N must all be vectors!')

    assert numel(grid_min)==numel(grid_max), 'grid min and grid_max must have the same number of elements!'

    assert numel(grid_min)== numel(N), 'grid min, grid_max, and N must have the same number of elements!'

    grid_min = to_column_mat(grid_min)
    grid_max = to_column_mat(grid_max)
    N = to_column_mat(N)

    # Create the grid
    g = Bundle(dict(
                    dim=len(grid_min), min=grid_min,
                    max=grid_max, N=N, bdry= cell(len(grid_min), 1)
                    ))

    #axis to ignore for target set creation
    g.axis_align = pdDims

    for i in range(g.dim):
        if np.any(i == pdDims):
            g.bdry[i] = addGhostPeriodic
        else:
            g.bdry[i] = addGhostExtrapolate

    if process:
      g = processGrid(g)
      
    return g
