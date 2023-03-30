__author__ 		= "Lekan Molu"
__copyright__ 	= "2021, Hamilton-Jacobi Analysis in Python"
__license__ 	= "Molux Licence"
__maintainer__ 	= "Lekan Molu"
__email__ 		= "patlekno@icloud.com"
__status__ 		= "Testing"

import numpy as np
import matplotlib.pyplot as plt
from os.path import  join


def show2D(g, mesh, title='', winsize=(16,9), ec='k', disp=False,
            fc='c', ax=None, savedict=None,  level=0):
    """
     show2D: display a 2D implicit surface function.

     Example: show2D(g, mesh, **kwargs)

     2D implicit surface functions can be displayed either by contour plot
       or by surface plot.  This routine does both.

     parameters:
       g   	Grid structure (see processGrid.m for details).
       mesh Array containing the implicit surface function.
       level: level set of the value function to plot.

       This function would substitute nicely for Sylvia's visFuncIm is g.dim<2

       Lekan Molu, September 07, 2021
   """
    if not savedict: savedict = {"save": False, "savepath": join("..", "jpeg_dumps")}

    if not ax: # if no gfigure given, gene figure
        fig = plt.figure(figsize=winsize)
        ax = fig.add_subplot(131, projection='3d')

    fontdict = {'fontsize':12, 'fontweight':'bold'}

    if g.dim<2:
        ax.plot(g.xs[0],  np.squeeze(mesh), linewidth=2, color=fc)
        if disp: plt.show()
        return

    ax.plot_surface(g.xs[0], g.xs[1], mesh, rstride=1, cstride=1,
                    cmap='viridis', edgecolor=ec, facecolor=fc)
    ax.set_xlabel('X', fontdict=fontdict)
    ax.set_ylabel('Y', fontdict=fontdict)
    ax.set_zlabel('Z', fontdict=fontdict)
    ax.set_title(f'{title} Mesh Surface')


    ax = fig.add_subplot(132,projection='3d')
    ax.contourf(g.xs[0], g.xs[1], mesh, colors=fc)
    ax.set_xlabel('X', fontdict=fontdict)
    ax.set_ylabel('Y', fontdict=fontdict)
    ax.set_zlabel('Z', fontdict=fontdict)
    ax.set_title(f'Contours')

    ax = fig.add_subplot(133)
    ax.contour(g.xs[0], g.xs[1], mesh, levels=1, colors=fc)
    ax.set_xlabel('X', fontdict=fontdict)
    ax.set_ylabel('Y', fontdict=fontdict)
    ax.grid('on')
    ax.set_title(f'2D level set')

    if savedict["save"]:
        fig.savefig(join(savedict["savepath"],savedict["savename"]),
                    bbox_inches='tight',facecolor='None')
    if disp:
        plt.show()
