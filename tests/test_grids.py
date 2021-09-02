from math import pi
from Utilities import *
from Grids import *
from BoundaryCondition import *
from Visualization import Visualizer
from ValueFuncs import proj


def test_create():
    'Tests grids creation'
    grid_min = expand(np.array((-5, -5, -pi)), ax = 1); # Lower corner of computation domain
    grid_max = expand(np.array((5, 5, pi)), ax = 1);   # Upper corner of computation domain
    N = expand(np.array((41, 41,  41)), ax = 1);        # Number of grid points per dimension
    pdDims = 3;               # 3rd dimension is periodic
    g = createGrid(grid_min, grid_max, N, pdDims);

    gridOut = processGrid(g)

def test_sep(num_points=45, low_mem=True, process=False):
    'Tests grids separation'
    gridIn=expand(np.array((0, 1, 0, 1)), 1)
    gridOut =expand(np.array((1, 2, 1, 2)), 1)
    N = num_points*ones(4,1).astype(np.int64)

    g = createGrid(gridIn, gridOut, N, process, low_mem);

    # print(f'len(g.xs), g.xs[0].shape {len(g.xs), g.xs[0].shape} g.N {g.N.shape}')
    dims = [[0, 2], [1, 3]]

    gs = sepGrid(g, dims);
    # print(f'len(gs[0].xs), gs[0].xs[0].shape {len(gs[0].xs), gs[0].xs[0].shape}')
    # print(f'len(gs[1].xs), gs[1].xs[1].shape {len(gs[1].xs), gs[1].xs[0].shape}')

    return g, gs

if __name__ == '__main__':

    viz = Visualizer(winsize=(16, 16))
    print('>>>>>>>>>>Testing creator >>>>>>>>>>>>>>>')
    g = test_create()
    viz.visGrid(g, dim= len(g))

    print('>>>>>>>>>>>Separating created grid>>>>>>>>')
    g, gs = test_sep(num_points=45, low_mem=False, process=True)
    viz.visGrid(gs, dim= len(gs))
    print('###########Successfully finished Grids test#########')
