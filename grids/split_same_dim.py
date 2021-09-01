import numpy as np
from utils import size, ind2sub
import .createGrid

def splitGrid_sameDim(g, bounds, padding):
    """
     gs = splitGrid_sameDim(g, bounds, padding)
         Splits the grid into smaller grids, each with specified bounds.
         Optionally, padding can be specified so that the grids overlap

     Inputs:
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

    Status: Under development. Use Sep Grids for now
     """

    assert isinstance(bounds, list), 'bounds must be a list or list of lists'
    ## Create a grid for the bounds
    if g.dim > 1:
        bounds_grid = np.meshgrid(*bounds, sparse=False);
    else:
        bounds_grid = np.meshgrid(bounds)[0]

    ## Create grids based on the bound grid
    bds_grd = bounds_grid[0]
    bds_grd_shp = np.array(bds_grd.shape)

    gs = bds_grd_shp-(bds_grd_shp>1)
    gs = np.empty(gs)

    for i in range(numel(gs)):
        ii = np.unravel_index(size(gs), i, order='F')
        # iip = ii;
        # for j in range(g.dim):
        #     iip[j] = iip[j] + 1;
        grid_min = bounds_grid[0]
        grid_max = bounds_grid[0]
        for j in range(1, g.dim):
           grid_min = np.concatenate(grid_min, bounds_grid[j][ii.flatten()], 0)
           grid_max = np.concatenate(grid_max, bounds_grid[j][iip.flatten()], 0)

        grid_min, grid_max, N = getOGPBounds(g, grid_min, grid_max, padding);

        gs[ii.flatten()] = createGrid(grid_min, grid_max, N);

    return gs
