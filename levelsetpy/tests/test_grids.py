__author__ 		= "Lekan Molu"
__copyright__ 	= "2021, Hamilton-Jacobi Analysis in Python"
__credits__  	= "There are None."
__license__ 	= "MIT License"
__maintainer__ 	= "Lekan Molu"
__email__ 		= "patlekno@icloud.com"
__status__ 		= "Completed"


import time
import pytest 
import os, sys
import numpy as np
from os.path import abspath, join, expanduser

sys.path.append(abspath(join('..')))

from math import pi

from levelsetpy.grids import *
import matplotlib.pyplot as plt
from levelsetpy.utilities import *
from levelsetpy.boundarycondition import *
from levelsetpy.visualization import *
from math import pi
from levelsetpy.initialconditions import *


@pytest.fixture(scope="class")
def grid_config():
    delay = 1e-3
    block=False
    fontdict = {'fontsize':12, 'fontweight':'bold'}

    path = join(expanduser("~"), "Downloads/leveltests")
    os.makedirs(path, exist_ok=True)

    savedict = dict(save=True, savename="generic", savepath=join(path))

    return (delay, block, fontdict, path, savedict)

@pytest.fixture
def grid_2d():
    gridMin = np.array([[0,0]])
    gridMax = np.array([[5, 5]])
    N = 20 *np.ones((2,1)).astype(np.int64)
    g = createGrid(gridMin, gridMax, N, low_mem=False, process=True)

    return g 

@pytest.fixture
def grid_2d_pads():
    # A 2D grid
    g = createGrid(np.array([[0, 0]]).T, np.array([[1, 1]]).T, np.array([[101, 101]]).T);
    bounds = [[0, 0.5, 1], [0, 0.25, 0.75, 1]]
    padding = np.array([[0, 0]]).T

    return (g, bounds, padding)

@pytest.fixture 
def grid_3d():
    grid_min = expand(np.array((-5, -5, -pi)), ax = 1); # Lower corner of computation domain
    grid_max = expand(np.array((5, 5, pi)), ax = 1);   # Upper corner of computation domain
    N = 41*ones(3, 1).astype(np.int64)
    pdDims = 3;               # 3rd dimension is periodic
    g = createGrid(grid_min, grid_max, N, pdDims);

    return g

@pytest.fixture
def grid_4d():
    # ### A Basic 2-D Grid and a signed distance function cylinder
    num_points=30
    gridIn=expand(np.array((0, 1, 0, 1)), 1)
    gridOut =expand(np.array((1, 2, 1, 2)), 1)
    N = num_points*ones(4,1).astype(np.int64)
    g = createGrid(gridIn, gridOut, N, process=False, low_mem=True)

    return g 

@pytest.fixture
def grid_3d_subcells():
    gmin = zeros(3,1); gmax = ones(3,1); N = 75*ones(3,1)
    bounds = [[0, 0.33, 0.5, 0.8, 1], [0, 0.5, 0.75, 1], \
              np.linspace(0, 1, 5).tolist()]
    padding = zeros(3,1)
    g = createGrid(gmin, gmax, N)

    return (g, bounds, padding)
    
@pytest.fixture     
def viz_plot(grid, grid_config, savename, winsize=(16, 9)):
        delay, block, fontdict, base_path, savedict = grid_config
        savedict["savename"] = savename
        viz = Visualizer(winsize, block=block, savedict=savedict, fontdict=fontdict)
        viz.visGrid(grid, grid.dim, title='Simple 3D Grid')
        plt.pause(delay)
        plt.close()

        return savedict 

class TestGrids2d:
    def test_2d_viz(self, grid_2d, grid_config):
        delay, block, fontdict, base_path, savedict = grid_config        
        savedict["savename"]='2d_grid.jpg'

        viz = Visualizer(winsize=(8, 5), block=block, savedict=savedict, fontdict=fontdict)
        viz.visGrid([grid_2d], grid_2d.dim, title='Simple 2D Grid')
        plt.pause(delay)
        plt.close()
        
        assert os.path.exists(join(savedict["savepath"], savedict["savename"]))


    # # ### A 3-D Grid and a signed distance function cylinder
    # def test_3d_viz(self, grid_3d, grid_config):
    #     delay, block, fontdict, base_path, savedict = grid_config
    #     savedict["savename"]='3d_grid.jpg'

    #     viz = Visualizer(winsize=(16, 9), block=block, savedict=savedict, fontdict=fontdict)
    #     viz.visGrid(grid_3d, grid_3d.dim, title='Simple 3D Grid')
    #     plt.pause(delay)
    #     plt.close()
    #     assert os.path.exists(join(savedict["savepath"], savedict["savename"]))

    def test_split_grid(self, grid_2d_pads, grid_config):
        delay, block, fontdict, base_path, savedict = grid_config        
        g, bounds, padding = grid_2d_pads
        gs = splitGrid_sameDim(g, bounds, padding)
        savedict["savename"]=f'{g.dim}D_grid_{len(gs)}_cell.jpg'

        viz = Visualizer(winsize=(8, 5), block=block, savedict=savedict, fontdict=fontdict)
        viz.visGrid(gs, gs[0].dim, title=f'A {len(gs)}-cell Grid Example')
        plt.pause(delay)
        plt.close()
        assert os.path.exists(join(savedict["savepath"], savedict["savename"]))

    def test_grid_split(self, grid_3d_subcells, grid_config):
        delay, block, fontdict, base_path, savedict = grid_config
        # A 3D grid with subcells
        g, bounds, padding  = grid_3d_subcells
        gs = splitGrid_sameDim(g, bounds, padding)

        savedict["savename"]= f'{g.dim}D_grid_{len(gs)}_cell.jpg'
        fig = plt.figure(figsize=(16, 9))
        ax = fig.add_subplot(111,projection='3d')
        for i in range(len(gs)):
            g = gs[i]
            #viz.visGrid(gs[i], gs[i].dim, title=f'A {len(gs)}-cell {len(gs)}-Grid Example')
            ax.plot3D(g.xs[0].flatten(), g.xs[1].flatten(), g.xs[2].flatten())
        ax.set_title(f'A {len(gs)}-cell within a {gs[0].dim}-Grid Example')
        fig.savefig(join(savedict["savepath"],savedict["savename"]), bbox_inches='tight',facecolor='None')
        plt.pause(delay)
        plt.close()
        assert os.path.exists(join(savedict["savepath"], savedict["savename"]))

    # # Split into two; and project the split Grids back to 2D
    # def test_4d_split(self, g, grid_config):
    #     delay, block, fontdict, base_path = grid_config
    #     savedict = dict(save=True, savename='4d_grid.jpg', savepath=join(base_path))
        
    #     # this is same as sepGrid_test, but with low_mem=True and process=False to test the sepGrid function on a grid that has not been processed and is in low memory mode (i.e. without the xs attribute pre-computed)
    #     # print(f'len(g.xs), g.xs[0].shape {len(g.xs), g.xs[0].shape} g.N {g.N.shape}')
    #     dims = [[0, 2], [1, 3]]

    #     gs, ds = sepGrid(g, dims)

    #     # Visualize
    #     savedict["savename"] = f'{g.dim}D_grid_{len(gs)}_cell.jpg'
    #     viz = Visualizer(winsize=(8, 6), block=block, savedict=savedict)
    #     viz.visGrid(gs, dim= len(gs), dims=dims, title=f'A {len(gs)}-cell/{g.dim}D-Grid Example')

    #     plt.pause(delay)
    #     plt.close()

    # # 4 subcells grid
    # num_points=30
    # gridIn=expand(np.array((0, 1, 0, 1)), 1)
    # gridOut =expand(np.array((1, 2, 1, 2)), 1)
    # N = num_points*ones(4,1).astype(np.int64)
    # g = createGrid(gridIn, gridOut, N, process=True, low_mem=True);

    # # print(f'len(g.xs), g.xs[0].shape {len(g.xs), g.xs[0].shape} g.N {g.N.shape}')
    # dims = [[0, 2], [1, 2],  [1, 3], [0, 1]]

    # gs, dat = sepGrid(g, dims);

    # savedict["savename"] = f'{g.dim}D_grid_{len(gs)}_cell.jpg'
    # viz = Visualizer(winsize=(8, 5), block=block, savedict=savedict)
    # viz.visGrid(gs, len(dims), title=f'A {len(dims)}-cell Grid Example', dims=dims)
    # # plt.pause(delay)
    # # plt.close()

# # ### An Eight-Grid Cell
# #
# # + Be careful with the number of points here
# # + as too many points can cause memory issues

# # In[6]:


# ## Cells Division Example | Lekan August 05

# gridIn= expand(np.array((0, 1, 0, 1, 1, 2, 1, 2)), 1)
# gridOut =expand(np.array((1, 2, 1, 2, 2, 3, 2, 3)), 1)

# num_points = 10
# N = num_points*ones(8,1).astype(np.int64)

# g = createGrid(gridIn, gridOut, N, process=True);

# dims = [[0, 2], [1, 2],  [1, 3], [0, 1]]
# gs, data = sepGrid(g, dims);
# savedict["savename"] = f'{g.dim}D_grid_{len(gs)}_cell.jpg'
# viz = Visualizer(winsize=(8, 5), block=block, savedict=savedict)
# viz.visGrid(gs, len(gs), title=f'A {len(N)}-sub-grid Example', dims=dims)
# # plt.pause(delay)
# # plt.close()

# # ### Making cell partitions in Grids

# # In[7]:


# # Truncated grid

# # truncte test
# N = 101; gmin = -2*np.ones((2, 1), dtype=np.float64); gmax = 2*np.ones((2, 1), dtype=np.float64)
# g = createGrid(gmin, gmax, N, process=True)
# data = shapeRectangleByCorners(g, [-1, -1], [1, 1])
# savedict = {"save": True, 'savepath': join("..", "jpeg_dumps"), 'savename': '2d_rect_by_corners.jpg'}
# show2D(g, data, savedict=savedict, ec=None, disp=0, title="Original Grid")
# gNew, dataNew = truncateGrid(g, data, [0.5, 0.5], [1.5, 1.5]);
# show2D(gNew, dataNew, savedict=savedict, level=0, ec=None, disp=1, title='Truncated Grid')
