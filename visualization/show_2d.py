from .settings import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

def show2D(g, data, winsize=(16,9)):
    """
     show2D: display a 2D implicit surface function.

       show2D(g, data)

     2D implicit surface functions can be displayed either by contour plot
       or by surface plot.  This routine does both.

     parameters:
       g   	Grid structure (see processGrid.m for details).
       data        Array containing the implicit surface function.


    ---------------------------------------------------------------------------
     Set up two figures in which to place the results.
       Use the same figures every time.
   """
    f1 = fig = plt.figure(figsize=winsize)
    f2 = copy.copy(f1)
    # Contour plot of implicitly defined set.
    ax = f1.add_subplot(1, 1, 1)
    ax.contour3D(g.xs[0], g.xs[1], data, [ 0, 0 ], 'b-');
    ax.grid('on');
    ax.xaxis.set_tick_params(labelsize=labelsize)
    ax.yaxis.set_tick_params(labelsize=labelsize)
    # axis equal;
    # axis(g.axis);
    ax.set_xlabel('x', fontdict=fontdict)
    ax.set_ylabel('y', fontdict=fontdict)

    # Surface plot of implicit surface function.
    ax2 = f2.add_subplot(1, 1, 1)

    ax2.plot_surface(g.xs[0], g.xs[1], data);
    ax.grid('on');
    ax.xaxis.set_tick_params(labelsize=labelsize)
    ax.yaxis.set_tick_params(labelsize=labelsize)
    # axis equal;
    ax.grid('on');
    # axis(g.axis);
    ax.set_xlabel('x', fontdict=fontdict)
    ax.set_ylabel('y', fontdict=fontdict)
    ax.set_ylabel('\phi(x,y)', fontdict=fontdict)
