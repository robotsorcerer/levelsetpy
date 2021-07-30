import logging
import numpy as np
from utils import Bundle

logger = logging.getLogger(__name__)


def  createGrid(grid_min, grid_max, N, pdDims=None, process=True, low_mem=False):
    """
     g = createGrid(grid_min, grid_max, N, pdDim)

     Thin wrapper around processGrid to create a grid compatible with the
     level set toolbox

     Inputs:
       grid_min, grid_max - minimum and maximum bounds on computation domain
       N                  - number of grid points in each dimension
       pdDims             - periodic dimensions (eg. pdDims = [2 3] if 2nd and
                              3rd dimensions are periodic)
       process            - specifies whether to call processGrid to generate
                            grid points

     Output:
       g - grid structure

     Mo Chen, 2016-04-18
    """

    if not pdDims:
        pdDims = []

    # Input checks
    if not isinstance(N, np.ndarray):
        N = N*np.ones(grid_min.shape)

    if not (isinstance(grid_min, np.ndarray) or isinstance(grid_min, np.ndarray) or isinstance(grid_max, np.ndarray) or isinstance(N, np.ndarray)):
        logger.fatal('grid_min, grid_max, N must all be vectors!')

    assert len(grid_min)== len(grid_max), 'grid min and grid_max must have the same number of elements!'

    assert len(grid_min)== len(N), 'grid min, grid_max, and N must have the same number of elements!'

    def to_column_mat(A):
        n,m = A.shape
        if n<m:
            return A.T

    to_column_mat(grid_min)
    to_column_mat(grid_max);
    to_column_mat(N);

    # Create the grid
    g = Bundle(dict(
                    dim=len(grid_min), min=grid_min,
                    max=grid_max, N=N, bdry= [[] for i in range(len(grid_min))] #=(len(grid_min, 1))
    ))

    # g.bdry = cell(g.dim, 1);
    for i in range(g.dim):
        if np.any(i == pdDims):
            g.bdry[i] = addGhostPeriodic;
            g.max[i] = g.min[i] + (g.max[i] - g.min[i]) * (1 - 1/g.N[i]);
        else:
            g.bdry[i] = addGhostExtrapolate

    if low_mem:
      g.dx = np.divide(grid_max - grid_min, N-1)
      g.vs = [[] for i in range(g.dim)] #cell(g.dim, 1);
      for i in range(g.dim):
          g.vs[i] = expand(np.arange(grid_min[i],  grid_max[i],  g.dx[i]), 1)
    elif process:
      g = processGrid(g)
    return g
