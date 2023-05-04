__author__ 		= "Lekan Molu"
__copyright__ 	= "2021, Hamilton-Jacobi Analysis in Python"
__credits__  	= "There are None."
__license__ 	= "MIT License"
__maintainer__ 	= "Lekan Molu"
__email__ 		= "patlekno@icloud.com"
__status__ 		= "Completed"

import copy, os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from os.path import join
from LevelSetPy.Utilities import error
from .color_utils import cmaps
from mpl_toolkits.mplot3d import Axes3D

def buffered_axis_limits(amin, amax, buffer_factor=1.0):
    """
    Increases the range (amin, amax) by buffer_factor on each side
    and then rounds to precision of 1/10th min or max.
    Used for generating good plotting limits.
    For example (0, 100) with buffer factor 1.1 is buffered to (-10, 110)
    and then rounded to the nearest 10.
    """
    diff = amax - amin
    amin -= (buffer_factor-1)*diff
    amax += (buffer_factor-1)*diff
    magnitude = np.floor(np.log10(np.amax(np.abs((amin, amax)) + 1e-100)))
    precision = np.power(10, magnitude-1)
    amin = np.floor(amin/precision) * precision
    amax = np.ceil (amax/precision) * precision
    return (amin, amax)

class Visualizer():

    def __init__(self, fig=None, ax=None, winsize=None,
                labelsize=18, linewidth=6, fontdict=None,
                block=False, savedict=None):
        """
            Ad-hoc visualizer for Grids, and grid partitions

            fig: pyplot figure. If not passed, it's created
            ax: subfig of fig on which to plot figure
            winsize: size of pyplot window (default is (16,9))
            labelsize: size of plot x/y labels
            linewidth: width of 2D lines
            fontdict: fontweight and size for visualization
            block: whether to block screen after plot or not

            Author: Lekan MOlux, August/September 2021
        """
        if winsize is None:
            self.winsize =(16, 9)
        self.linewidth = linewidth
        if (fig and ax) is None:
            self._fig = plt.figure(figsize=winsize)
            self._fig.tight_layout()
        self.block=block

        self._labelsize = labelsize
        self._fontdict  = fontdict
        self._projtype = 'rectilinear'
        self.savedict = savedict

        if self.savedict["save"] and not os.path.exists(self.savedict["savepath"]):
            os.makedirs(self.savedict["savepath"])

        if self._fontdict is None:
            self._fontdict = {'fontsize':12, 'fontweight':'bold'}

    def visGrid(self, gs, dim, colors=None, save_dir=None, title=None, dims=None):
        # see helper OC/visualization/visGrid.m
        if not colors:
            colors = ['blue', 'red', 'yellow', 'orange', 'green', 'black']

        if dim==1:
            g = gs
            ax = self._fig.add_subplot(1, 1, 1)
            ax.plot(np.zeros((g.N, 1)), g.vs[0], color=colors[0], linestyle='.')
            ax.plot(np.hstack([g.min, g.max]), np.hstack([min(g.vs[0]), max(g.vs[0])]), \
                    linestyle='-', color=colors[0])
            ax.xaxis.set_tick_params(labelsize=self._labelsize)
            ax.yaxis.set_tick_params(labelsize=self._labelsize)
            if title:
                ax.set_title(title, fontdict=self._fontdict)

            if dims:
                x, y = dims[i]
                ax.annotate(f'dim={dims[i]}', xy=(x+0.25, y-0.5), size=17)

            self._fig.tight_layout()
            # if save_dir:
            #     self.fig.savefig(join(save_dir, datetime.strftime(datetime.now() + '%H-%')+'.png'))
        elif dim==2:
            ax = self._fig.add_subplot(1, 1, 1)

            title=f'A {len(gs)}-cell/{gs[0].dim}D-Grid'
            for i in range(len(gs)):
                g = gs[i]
                ax.plot(g.xs[0], g.xs[1], '.', color=colors[i])
                ax.plot(np.hstack([g.min[0], g.min[0]]), np.hstack([g.min[1], g.max[1]]), linestyle='-', color=colors[i])
                ax.plot(np.hstack([g.max[0], g.max[0]]), np.hstack([g.min[1], g.max[1]]), linestyle='-', color=colors[i])
                ax.plot(np.hstack([g.min[0], g.max[0]]), np.hstack([g.min[1], g.min[1]]), linestyle='-', color=colors[i])
                ax.plot(np.hstack([g.min[0], g.max[0]]), np.hstack([g.max[1], g.max[1]]), linestyle='-', color=colors[i])

                ax.xaxis.set_tick_params(labelsize=self._labelsize)
                ax.yaxis.set_tick_params(labelsize=self._labelsize)

                ax.set_xlabel('x', fontdict=self._fontdict)
                ax.set_ylabel('y', fontdict=self._fontdict)

                if dims:
                    x, y = dims[i]
                    ax.annotate(f'dim={dims[i]}', xy=(x+0.25, y-0.5), size=17)

            if not title:
                title = f'Gridsplitter along {len(gs)} dims'

            plt.title(title, fontdict=self._fontdict)
        elif dim==3:
            ax = self._fig.add_subplot(1, 2, 1, projection='3d')

            # title=f'A {len(gs)}-cell/{gs[0].dim}D-Grid'
            g = gs
            g.xs[0] = g.xs[0].reshape(np.prod(g.xs[0].shape), 1)
            g.xs[1] = g.xs[1].reshape(np.prod(g.xs[1].shape), 1)
            g.xs[2] = g.xs[2].reshape(np.prod(g.xs[2].shape), 1)

            ax.plot_wireframe(g.xs[0], g.xs[1], g.xs[2], rstride=10, cstride=10)

            ax.set_xlabel('x', fontdict=self._fontdict)
            ax.set_ylabel('y', fontdict=self._fontdict)
            ax.set_zlabel('z', fontdict=self._fontdict)

            ax.xaxis.set_tick_params(labelsize=self._labelsize)
            ax.yaxis.set_tick_params(labelsize=self._labelsize)
            ax.zaxis.set_tick_params(labelsize=self._labelsize)

            ax = self._fig.add_subplot(1, 2, 2, projection='3d')
            surf = ax.plot_surface(g.xs[0], g.xs[1], g.xs[2],rstride=1, cstride=1, cmap=cm.coolwarm,
                                   linewidth=0, antialiased=False)

            plt.title(title, fontdict=self._fontdict)
            self._fig.colorbar(surf, shrink=0.5, aspect=10)

        elif dim>3: # this is for a projected grid to 2d
            ax = self._fig.add_subplot(1, 1, 1)
            i=0
            for g in gs:
                ax.plot(g.xs[0], g.xs[1], '.', color=colors[i])
                ax.plot(np.hstack([g.min[0], g.min[0]]), np.hstack([g.min[1], g.max[1]]), linestyle='-', color=colors[i])
                ax.plot(np.hstack([g.max[0], g.max[0]]), np.hstack([g.min[1], g.max[1]]), linestyle='-', color=colors[i])
                ax.plot(np.hstack([g.min[0], g.max[0]]), np.hstack([g.min[1], g.min[1]]), linestyle='-', color=colors[i])
                ax.plot(np.hstack([g.min[0], g.max[0]]), np.hstack([g.max[1], g.max[1]]), linestyle='-', color=colors[i])

                ax.xaxis.set_tick_params(labelsize=self._labelsize)
                ax.yaxis.set_tick_params(labelsize=self._labelsize)

                ax.set_xlabel('x', fontdict=self._fontdict)
                ax.set_ylabel('y', fontdict=self._fontdict)
                if dims:
                    x, y = dims[i]
                    ax.annotate(f'dim={dims[i]}', xy=(x+0.25, y-0.5), size=17)
                i+=1
            title=f'A {len(gs)}-cell/{g.dim}D-Grid'
            self._fig.tight_layout()
            plt.title(title, fontdict=self._fontdict)
        else:
            error('Only Grids of up to 3 dimensions can be visualized!')

        if self.savedict["save"]:
            plt.savefig(join(self.savedict["savepath"],self.savedict["savename"]),
                        bbox_inches='tight', facecolor='None')

        plt.show(block=self.block)

    def visFuncIm(self, g, dataPlot,color):
        ax = self._fig.add_subplot(1, 1, 1)
        if g.dim<2:
            ax.plot(g.xs[0], np.squeeze(dataPlot), linewidth=2, color = color)
        elif g.dim==2:
            surf = ax.plot_surface(g.xs[0], g.xs[1], dataPlot, \
                                   rstride=1, cstride=1, cmap=cm.coolwarm,
                                   linewidth=0, antialiased=False, facecolors=color)
        else:
            error('Can not plot in more than 3D!')

        plt.show(block=self.block)
