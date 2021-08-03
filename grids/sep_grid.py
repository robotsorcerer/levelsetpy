import numpy as np
from utils import expand, ones
from grids import createGrid
from valFuncs import proj


def sepGrid(g, dims):
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
    gs = []
    #dims = [[0, 2], [1, 3]]
    for i in range(len(dims)):
        dims_i = ones(g.dim, 1);
        for j in dims[i]:
            dims_i[j, 0] = 0
        projection, dout = proj(g, np.empty((0, 0)), dims_i);
        gs.append(projection)

    return gs

def sepGridTest():
    gridIn=expand(np.array((0, 1, 0, 1)), 1)
    gridOut =expand(np.array((1, 2, 1, 2)), 1)
    N = 45*ones(4,1).astype(np.int64)

    g = createGrid(gridIn, gridOut, N);

    # print(f'len(g.xs), g.xs[0].shape {len(g.xs), g.xs[0].shape} g.N {g.N.shape}')
    dims = [[0, 2], [1, 3]]

    gs = sepGrid(g, dims);
    # print(f'len(gs[0].xs), gs[0].xs[0].shape {len(gs[0].xs), gs[0].xs[0].shape}')
    # print(f'len(gs[1].xs), gs[1].xs[1].shape {len(gs[1].xs), gs[1].xs[0].shape}')

    return gs
