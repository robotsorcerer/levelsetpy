
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from os.path import join
from utils import error

class Visualizer():

    def __init__(self, fig=None, ax=None, winsize=None,
                labelsize=18, linewidth=6, fontdict=None):
        """
            Ad-hoc visualizer for grids, grid partitions
            and HJI solutions

            Mimics Ian Mitchell's Visualization function
        """
        if winsize is None:
            self.winsize =(16, 9)
        self.linewidth = linewidth
        if (fig and ax) is None:
            self._fig = plt.figure(figsize=winsize)
            self._fig.tight_layout()

        self._labelsize = labelsize
        self._fontdict  = fontdict
        self._projtype = 'rectilinear'
        if self._fontdict is None:
            self._fontdict = {'fontsize':12, 'fontweight':'bold'}

    def visGrid(self, gs, dim, colors=None, save_dir=None):
        # see helper OC/visualization/visGrid.m
        if not colors:
            colors = ['blue', 'red', 'yellow', 'orange', 'green', 'black']

        if dim==1:
            g = gs
            ax = self._fig.add_subplot(1, 1, 1)
            ax.plot(np.zeros((g.N, 1)), g.vs[0], color=colors[0], linestyle='.')
            ax.plot(np.hstack([g.min, g,max]), np.hstack([min(g.vs[0]), max(g.vs[0])]), \
                    linestyle='-', color=colors[0])
            ax.xaxis.set_tick_params(labelsize=self._labelsize)
            ax.yaxis.set_tick_params(labelsize=self._labelsize)

            self._fig.tight_layout()
            # if save_dir:
            #     self.fig.savefig(join(save_dir, datetime.strftime(datetime.now() + '%H-%')+'.png'))
            plt.show()
        elif dim==2:
            ax = self._fig.add_subplot(1, 1, 1)

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

            plt.title(f'Gridsplitter along {len(gs)} dims')
            plt.show()

        elif dim==3:
            ax = self._fig.add_subplot(1, 2, 1, projection='3d')

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
            self._fig.colorbar(surf, shrink=0.5, aspect=10)

            plt.show()
        else:
            error('Only grids of up to 3 dimensions can be visualized!')


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
