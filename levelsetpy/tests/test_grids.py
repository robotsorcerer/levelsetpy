__author__ 		= "Lekan Molu"
__copyright__ 	= "2021, Hamilton-Jacobi Analysis in Python"
__credits__  	= "There are None."
__license__ 	= "MIT License"
__maintainer__ 	= "Lekan Molu"
__email__ 		= "patlekno@icloud.com"
__status__ 		= "Completed"


import time
import os, sys
import numpy as np
from os.path import abspath, join

sys.path.append(abspath(join('..')))

# filepath = os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) )
# print("filepath: ", filepath)
# sys.path.append( filepath )

from levelsetpy.grids import *
import matplotlib.pyplot as plt
from levelsetpy.utilities import *
from levelsetpy.boundarycondition import *
from levelsetpy.visualization import *
from math import pi
from levelsetpy.initialconditions import *


# ### A Basic 2-D Grid and a signed distance function cylinder

# In[2]:
delay = 1
block=False
fontdict = {'fontsize':12, 'fontweight':'bold'}

from math import pi
gridMin = np.array([[0,0]])
gridMax = np.array([[5, 5]])
N = 20 *np.ones((2,1)).astype(np.int64)
g = createGrid(gridMin, gridMax, N, low_mem=False, process=True)


savedict = dict(save=True, savename='2d_grid.jpg', savepath=join("..", "jpeg_dumps"))
viz = Visualizer(winsize=(8, 5), block=block, savedict=savedict)
viz.visGrid([g], g.dim, title='Simple 2D Grid')
# plt.pause(delay)
# plt.close()

# ### A 3-D Grid and a signed distance function cylinder

# In[3]:

grid_min = expand(np.array((-5, -5, -pi)), ax = 1); # Lower corner of computation domain
grid_max = expand(np.array((5, 5, pi)), ax = 1);   # Upper corner of computation domain
N = 41*ones(3, 1).astype(np.int64)
pdDims = 3;               # 3rd dimension is periodic
g = createGrid(grid_min, grid_max, N, pdDims);

data0 = shapeCylinder(g, 3, zeros(len(N), 1), radius=1)

savedict["savename"] = '3d_grid.jpg'
viz = Visualizer(winsize=(16, 9), block=block, savedict=savedict)
viz.visGrid(g, g.dim, title='Simple 3D Grid')
# plt.pause(delay)
# plt.close()

# ### A 4-D Grid
#
# Split into two; and project the split Grids back to 2D

# In[4]:


# this is same as sepGrid_test
num_points=30
gridIn=expand(np.array((0, 1, 0, 1)), 1)
gridOut =expand(np.array((1, 2, 1, 2)), 1)
N = num_points*ones(4,1).astype(np.int64)
g = createGrid(gridIn, gridOut, N, process=False, low_mem=True);

# print(f'len(g.xs), g.xs[0].shape {len(g.xs), g.xs[0].shape} g.N {g.N.shape}')
dims = [[0, 2], [1, 3]]

# gs, ds = sepGrid(g, dims)

# # Visualize
# savedict["savename"] = f'{g.dim}D_grid_{len(gs)}_cell.jpg'
# viz = Visualizer(winsize=(8, 6), block=block, savedict=savedict)
# viz.visGrid(gs, dim= len(gs), dims=dims, title=f'A {len(gs)}-cell/{g.dim}D-Grid Example')

# plt.pause(delay)
# plt.close()

# ### A 4-D Grid Split into 4 subGrids;
#
# + re-projected the split Grids back to 2D

# In[5]:


# 4 subcells grid
num_points=30
gridIn=expand(np.array((0, 1, 0, 1)), 1)
gridOut =expand(np.array((1, 2, 1, 2)), 1)
N = num_points*ones(4,1).astype(np.int64)
g = createGrid(gridIn, gridOut, N, process=True, low_mem=True);

# print(f'len(g.xs), g.xs[0].shape {len(g.xs), g.xs[0].shape} g.N {g.N.shape}')
dims = [[0, 2], [1, 2],  [1, 3], [0, 1]]

gs, dat = sepGrid(g, dims);

savedict["savename"] = f'{g.dim}D_grid_{len(gs)}_cell.jpg'
viz = Visualizer(winsize=(8, 5), block=block, savedict=savedict)
viz.visGrid(gs, len(dims), title=f'A {len(dims)}-cell Grid Example', dims=dims)
# plt.pause(delay)
# plt.close()

# ### An Eight-Grid Cell
#
# + Be careful with the number of points here
# + as too many points can cause memory issues

# In[6]:


## Cells Division Example | Lekan August 05

gridIn= expand(np.array((0, 1, 0, 1, 1, 2, 1, 2)), 1)
gridOut =expand(np.array((1, 2, 1, 2, 2, 3, 2, 3)), 1)

num_points = 10
N = num_points*ones(8,1).astype(np.int64)

g = createGrid(gridIn, gridOut, N, process=True);

dims = [[0, 2], [1, 2],  [1, 3], [0, 1]]
gs, data = sepGrid(g, dims);
savedict["savename"] = f'{g.dim}D_grid_{len(gs)}_cell.jpg'
viz = Visualizer(winsize=(8, 5), block=block, savedict=savedict)
viz.visGrid(gs, len(gs), title=f'A {len(N)}-sub-grid Example', dims=dims)
# plt.pause(delay)
# plt.close()

# ### Making cell partitions in Grids

# In[7]:


# A 2D grid
g = createGrid(np.array([[0, 0]]).T, np.array([[1, 1]]).T, np.array([[101, 101]]).T);

bounds = [[0, 0.5, 1], [0, 0.25, 0.75, 1]]
padding = np.array([[0, 0]]).T;
gs = splitGrid_sameDim(g, bounds, padding);

savedict["savename"] = f'{g.dim}D_grid_{len(gs)}_cell.jpg'
viz = Visualizer(winsize=(8, 5), block=block, savedict=savedict)
viz.visGrid(gs, gs[0].dim, title=f'A {len(gs)}-cell Grid Example')
# plt.pause(delay)
# plt.close()

# ### A 3D grid with subcells

# In[8]:


# A 3D grid
gmin = zeros(3,1); gmax = ones(3,1); N = 75*ones(3,1)
bounds = [[0, 0.33, 0.5, 0.8, 1], [0, 0.5, 0.75, 1], np.linspace(0, 1, 5).tolist()]
padding = zeros(3,1)
g = createGrid(gmin, gmax, N);
gs = splitGrid_sameDim(g, bounds, padding);

# In[9]:
savedict["savename"] = f'{g.dim}D_grid_{len(gs)}_cell.jpg'
fig = plt.figure(figsize=(16, 9))
ax = fig.add_subplot(111,projection='3d')
for i in range(len(gs)):
    g = gs[i]
    #viz.visGrid(gs[i], gs[i].dim, title=f'A {len(gs)}-cell {len(gs)}-Grid Example')
    ax.plot3D(g.xs[0].flatten(), g.xs[1].flatten(), g.xs[2].flatten())
ax.set_title(f'A {len(gs)}-cell within a {gs[0].dim}-Grid Example')
fig.savefig(join(savedict["savepath"],savedict["savename"]), bbox_inches='tight',facecolor='None')

plt.show()
# plt.pause(delay)
plt.close()


# Truncated grid

# truncte test
N = 101; gmin = -2*np.ones((2, 1), dtype=np.float64); gmax = 2*np.ones((2, 1), dtype=np.float64)
g = createGrid(gmin, gmax, N, process=True)
data = shapeRectangleByCorners(g, [-1, -1], [1, 1])
savedict = {"save": True, 'savepath': join("..", "jpeg_dumps"), 'savename': '2d_rect_by_corners.jpg'}
show2D(g, data, savedict=savedict, ec=None, disp=0, title="Original Grid")
gNew, dataNew = truncateGrid(g, data, [0.5, 0.5], [1.5, 1.5]);
show2D(gNew, dataNew, savedict=savedict, level=0, ec=None, disp=1, title='Truncated Grid')
