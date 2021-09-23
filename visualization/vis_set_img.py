from Utilities import *
from Grids import *
from .show_3d import show3D
from ValueFuncs import proj
import matplotlib.pyplot as plt

def visSetIm(data, g=None, ax=None, color='r', level=0, extraArgs=None):
    """
     h = visSetIm(g, data, color, level, sliceDim,)
     Code for quickly visualizing level sets

     Inputs: g          - grid structure
             data       - value function corresponding to grid g
             color      - (defaults to red)
             level      - level set to display (defaults to 0)
             sliceDim   - for 4D sets, choose the dimension of the slices (defaults
                          to last dimension)
    """
    ## Default parameters and input check
    if g is None:
        N = np.asarray(size(data), order=ORDER_TYPE).T
        g = createGrid(np.ones(data.ndim, 1, order=ORDER_TYPE), N, N)

    if ax is None:
        fig = plt.figure(figsize=(12,7))
        ax = fig.add_subplot(111)

    if g.dim != numDims(data) and g.dim+1 != numDims(data):
        error('Grid dimension is inconsistent with data dimension!')

    if not extraArgs:
        extraArgs = Bundle({'disp':False})

    levels = extraArgs.levels if isfield(extraArgs, 'levels') else 0

    if isfield(extraArgs, 'fontdict'):
        fontdict = extraArgs.fontdict
    else:
         fontdict = {'fontsize':12, 'fontweight':'bold'}
         extraArgs.fontdict=fontdict

    if isfield(extraArgs, 'title'):
        title = extraArgs.title
    else:
        title = f'2D level set'
        extraArgs.title = title

    ax.contour(g.xs[0], g.xs[1], data, levels=levels, colors=color)
    ax.set_xlabel('X', fontdict=fontdict)
    ax.set_ylabel('Y', fontdict=fontdict)
    ax.grid('on')
    ax.set_title(title)

    # save_png = False
    # if isfield(extraArgs, 'fig_filename'):
    #     save_png = True
    #     fig_filename = extraArgs.fig_filename
    if g.dim == numDims(data):
        # Visualize a single set
        visSetIm_single(g, data, ax, color, level, extraArgs)
    else:
      dataSize = size(data)
      numSets = dataSize[-1]

      # fig = plt.figure(figsize=(16,9))
      # fig.tight_layout()
      # ax = self._fig.add_subplot(1, 1, 1)
      # extraArgs.ax = ax

      for i in range(numSets):
        if i>1:
            ax.cla()
            visSetIm_single(g, data[i,...], ax, color, level, extraArgs)
        elif i == 1:
            visSetIm_single(g, data[i,...], ax, color, level, extraArgs)

## Visualize a single set
def visSetIm_single(g, data, ax, color, level, extraArgs):

    sliceDim = g.dim # Slice last dimension by default

    if g.dim==1:
        ax.plot(g.xs[0], data, linestyle='-', color=color)
        ax.plot(g.xs[0], np.zeros(size(g.xs[0])), linestyle=':', color='k')

    elif g.dim==2:
        # show2D(g, data, ax=ax, fc='g', savedict = {"save": False}, disp=extraArgs.disp)
        # ax = fig.add_subplot(133)
        ax.contour(g.xs[0], g.xs[1], data, levels=level, colors=color)
        ax.set_xlabel('X', fontdict=extraArgs.fontdict)
        ax.set_ylabel('Y', fontdict=extraArgs.fontdict)
        ax.grid('on')
        # ax.set_title(f'2D level set')

    elif g.dim==3:
        show3D(g, data, fc='g', savedict = {"save": False}, disp=extraArgs.disp)
    elif g.dim==4:
        visSetIm4D(g, data, color, level, sliceDim, applyLight)

## 3D Visualization
def visSetIm3D( ax, g, data, color, level=0., disp=True):
    show3D(g, data, fc=color, level=level, disp=disp)

## 4D Visualization
def visSetIm4D(g, data, color, level, sliceDim, disp=True):
    # Takes 6 slices in the dimension sliceDim and shows the 3D projections
    N = 6
    spC = 3
    spR = 2
    fig = plt.figure(figsize=(16, 9))
    for i in range(N):
        ax = fig.add_subplot(spR, spC, i, projection='3d')
        xs = g.min[sliceDim] + i/(N+1) * (g.max[sliceDim] - g.min[sliceDim])

        dim = np.zeros(([4,1]), order=ORDER_TYPE)
        dim[sliceDim, 0] = 1
        g3D, data3D = proj(g, data, dim, xs)

        # Visualize 3D slices
        show3D(g3D, data3D, color, ax=ax, level=level, disp=disp)
    plt.show()
