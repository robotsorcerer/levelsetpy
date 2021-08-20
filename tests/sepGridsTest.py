import numpy as np
from utils import expand, ones
from grids import createGrid, sepGrid
from valFuncs import proj


def sepGridTest(num_points=45, low_mem=True, process=False):
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
