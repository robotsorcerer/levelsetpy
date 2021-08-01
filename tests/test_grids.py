from math import pi
from numpy import np
from utils import expand
from grids import createGrid, processGrid

def test_create():
    grid_min = expand(np.array((-5, -5, -pi)), ax = 1); # Lower corner of computation domain
    grid_max = expand(np.array((5, 5, pi)), ax = 1);   # Upper corner of computation domain
    N = expand(np.array((41, 41,  41)), ax = 1);        # Number of grid points per dimension
    pdDims = 3;               # 3rd dimension is periodic
    g = createGrid(grid_min, grid_max, N, pdDims);

    gridOut = processGrid(g)

def test_sep():
    
