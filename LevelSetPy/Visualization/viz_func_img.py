import numpy as np
import matplotlib.pyplot as plt

def visFuncIm(gPlot,dataPlot,ax, color,disp=False):

    fontdict = {'fontsize':12, 'fontweight':'bold'}
    if ax is None:
        fig = plt.figure(figsize=(12,7))
        ax = fig.add_subplot(111)

    if gPlot.dim<2:
        plt.plot(gPlot.xs[0], np.squeeze(dataPlot), color=color, linewidth=2);
    elif gPlot.dim==2:
        ax.plot_surface(gPlot.xs[0], gPlot.xs[1], dataPlot, rstride=1, cstride=1,
                        cmap='viridis', edgecolor='k', facecolor=color)
        ax.set_xlabel('X', fontdict=fontdict)
        ax.set_ylabel('Y', fontdict=fontdict)
        ax.set_zlabel('Z', fontdict=fontdict)
        ax.set_title(f'Value Func Surface')
    else:
        error('Can not plot in more than 3D!')
    if disp:
        plt.show()

    return ax
