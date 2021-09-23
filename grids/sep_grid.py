import numpy as np
from Utilities import expand, ones, ORDER_TYPE
from .create_grid import createGrid
from ValueFuncs import proj


def sepGrid(g, dims, data=np.empty((0, 0))):
    """
        gs = sepGrid(g, dims)
       Separates a grid into the different dimensions specified in dims

     Inputs:
       g    - grid
       dims - cell structure of grid dimensions
                eg. {[1 3], [2 4]} would split the grid into two; one grid in
                    the 1st and 3rd dimensions, and another in the 2nd and 4th
                    dimensions

     Output:
       gs - cell vector of separated grids
   """
    gs, ds = [], []
    #dims = [[0, 2], [1, 3]]
    for i in range(len(dims)):
        dims_i = ones(g.dim, 1, order=ORDER_TYPE).astype(np.int64);
        for j in dims[i]:
            dims_i[j, 0] = 0
        projection, dout = proj(g, data, dims_i);
        gs.append(projection)
        ds.append(dout)

    return gs, ds
