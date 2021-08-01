from utils import ones
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
    gs = [[] for  i in range(len(dims))]
    #dims = [[0, 2], [1, 3]]
    for i in range(len(dims)):
        dims_i = ones(1, g.dim);
        for j in dims[i]:
            dims_i[0, j] = 0
        gs[i] = proj(g, None, dims_i);

    return gs
