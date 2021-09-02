import numpy as np
import matplotlib.pyplot as plt

def visFuncIm(gPlot,dataPlot,color,alpha):

    if gPlot.dim<2:
        h = plt.plot(gPlot.xs[0], np.squeeze(dataPlot), linewidth=2);
        h.Color = color;
    elif gPlot.dim==2:
        h = plt.surf(gPlot.xs[0], gPlot.xs[1], dataPlot);
        h.EdgeColor = 'none';
        h.FaceColor = color;
        h.FaceAlpha = alpha;
        h.FaceLighting = 'phong';
    else:
        error('Can not plot in more than 3D!')

    return h
