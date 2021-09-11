from Utilities import error

def hit_edge3d(out, x, y, z):
    """
        Returns True if we hit the edge of a 3D grid
        while searching for cell neighbors.

        Author: Lekan Molu, September 05, 2021
    """
    return ((x<0) or (y<0) or (z<0) or (x>=out[0]) or (y>=out[1]) or (z>=out[2]))

def hit_edge2d(out, r, c):
    """
        Returns True if we hit the edge of a 2D grid
        while searching for cell neighbors.

        Author: Lekan Molu, September 05, 2021
    """
    return ((r<0) or (c<0) or (r>=out[0]) or (c>=out[1]))

def neighbors(idx, out_shape):
    if len(idx)==2:
        return neighs2d(idx, out_shape)
    elif len(idx) == 3:
        return neighs3d(idx, out_shape)
    else:
        error("Neighbors of only 2d and 3d grids are currently accounted for")

def neighs2d(idx, out_shape):
    '''
        Inputs:
            idx: index of the cell within the 2D grid.
            out_shape: shape of the containing grid for all the cells.
        Output:
            Returns the indices of a cell as an array of tuples.

        Find the neighbors of a 2D cell within the containing grid.
        At any given time, we are looking in no more than len(idx)*2 directions.
        In 2D for example, we will iteratively look to the [left], [right], [top],
        and [bottom] and determine the neighbors provided that we are not at an edge.
        If we happen to look beyond the bounds of the grid containing the cell (i.e. we
        are at an edge), we'll discard the results we've "mistakenly" added.

        This is faster than for loops and is actually O(n).

        Author: Lekan Molu, September 05, 2021.
    '''
    r, c = idx[0], idx[1]

    assert r<out_shape[0], f'Error at idx: {idx}; cell\'s row index cannot be greater than encapsulating grid\'s row index'
    assert c<out_shape[1], f'Error at idx: {idx}; cell\'s column index cannot be greater than encapsulating grid\'s column index'

    direc = 0
    dr = [-1, 2, -1, 0]
    dc = [0,  0, -1, 2]

    res = []
    while direc<len(idx)*2:
        r += dr[direc]
        c += dc[direc]

        res.append((r,c))
        if hit_edge2d(out_shape, r, c):
            res = res[:-1]
        direc += 1

    return res

def neighs3d(idx, out_shape):
    '''
        Inputs:
            idx: Tuple representing index of the cell within the 3D grid.
            out_shape: Tuple representing the shape of the containing grid for all the cells.
        Output:
            Returns the indices of a cell as an array of tuples.

        Find the neighbors of a 3D cell within the containing grid.
        At any given time, we are looking in no more than len(idx)*3 directions.
        In 3D for example, we will iteratively look to the [+x], [-x], [+y], [-y], [+z],
        -[z] directions, determine the neighbors provided that we are not at an edge.
        If we happen to look beyond the bounds of the grid encapsulating a cell (i.e. we
        are at an edge), we'll discard the results we've "mistakenly" added.

        This is faster than for loops and is actually O(n).

        Author: Lekan Molu, September 05, 2021.
    '''
    x, y, z = idx

    assert x<out_shape[0], 'cell\'s x index cannot be greater than encapsulating grid\'s x index'
    assert y<out_shape[1], 'cell\'s y index cannot be greater than encapsulating grid\'s y index'
    assert z<out_shape[2], 'cell\'s z index cannot be greater than encapsulating grid\'s z index'

    direc = 0
    # Go (x-1, y, z), (x+1, y, z), (x, y-1, z), (x, y+1, z), (x, y, z-1), (x, y, z+1)
    dx = [-1, 2, -1, 0,  0, 0]
    dy = [0,  0, -1, 2, -1, 0]
    dz = [0,  0,  0, 0, -1, 2]

    res = []
    while direc<len(idx)*2:
        x += dx[direc]
        y += dy[direc]
        z += dz[direc]

        res.append((x, y, z))
        if hit_edge3d(out_shape, x, y, z):
            res = res[:-1]
        direc += 1

    return res
