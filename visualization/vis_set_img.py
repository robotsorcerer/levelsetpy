from Utilities import *
from Grids import *

def visSetIm(data, g=None, color='r', level=0, extraArgs=None):
    """
     h = visSetIm(g, data, color, level, sliceDim, applyLight)
     Code for quickly visualizing level sets

     Inputs: g          - grid structure
             data       - value function corresponding to grid g
             color      - (defaults to red)
             level      - level set to display (defaults to 0)
             sliceDim   - for 4D sets, choose the dimension of the slices (defaults
                          to last dimension)
             applyLight - Whether to apply camlight (defaults to True)

     Output: h - figure handle

     Adapted from Ian Mitchell's visualizeLevelSet function from the level set
     toolbox

     Mo Chen, 2016-05-12
    """
    ## Default parameters and input check
    if g is None:
      N = np.asarray(size(data)).T
      g = createGrid(np.ones(numDims(data), 1), N, N)

    # print('g: ', g)
    if g.dim != numDims(data) and g.dim+1 != numDims(data):
      error('Grid dimension is inconsistent with data dimension!')

    if not extraArgs:
      extraArgs = Bundle({})

    deleteLastPlot = True
    if isfield(extraArgs, 'deleteLastPlot'):
      deleteLastPlot = extraArgs.deleteLastPlot

    save_png = False
    if isfield(extraArgs, 'fig_filename'):
      save_png = True
      fig_filename = extraArgs.fig_filename
    ##
    if g.dim == numDims(data):
      # Visualize a single set
      h = visSetIm_single(g, data, color, level, extraArgs)

    else:
      dataSize = size(data)
      numSets = dataSize[-1]

      fig = plt.figure(figsize=(16,9))
      fig.tight_layout()
      ax = self._fig.add_subplot(1, 1, 1)
      extraArgs.ax = ax

      for i in range(numSets):
        if i > 1:
          extraArgs.applyLight = False

        if deleteLastPlot:
          if i > 1:
            del h
          h = visSetIm_single(g, data[i,...], color, level, extraArgs)
        else:
          if i == 1:
            h = cell(numSets, 1)

          h[i] = visSetIm_single(g, data[i,...], color, level, extraArgs)

## Visualize a single set
def visSetIm_single(g, data, color, level, extraArgs):

    sliceDim = g.dim # Slice last dimension by default
    applyLight = True # Add cam light by default
    LineStyle = '-'
    LineWidth = 1

    if isfield(extraArgs, 'sliceDim'):
        sliceDim = extraArgs.sliceDim

    if isfield(extraArgs, 'applyLight'):
        applyLight = extraArgs.applyLight

    if isfield(extraArgs, 'LineStyle'):
        LineStyle = extraArgs.LineStyle

    if isfield(extraArgs, 'LineWidth'):
        LineWidth = extraArgs.LineWidth

    if g.dim==1:
        extraArgs.ax.plot(g.xs[0], data, linestyle='-', color=color)
        extraArgs.ax.plot(g.xs[0], np.zeros(size(g.xs[0])), linestyle=':', color='k')

    elif g.dim==2:
        if level:
          [~, h] = ax.contour(g.xs[0], g.xs[1], data, [level, level], 'color', color)
        elseif isempty(level):
          [~, h] = ax.contour(g.xs[0], g.xs[1], data)
        else
          [~, h] = ax.contour(g.xs[0], g.xs[1], data, level, 'color', color)

        h.LineStyle = LineStyle
        h.LineWidth = LineWidth
    # elif g.dim==3:
    #     h = visSetIm3D(g, data, color, level, applyLight)
    #
    # elif g.dim==4:
    #     h = visSetIm4D(g, data, color, level, sliceDim, applyLight)
    #
    # ## 3D Visualization
    # function h = visSetIm3D(g, data, color, level, applyLight)
    # # h = visSetIm3D(g, data, color, level, applyLight)
    # # Visualizes a 3D reachable set
    #
    #
    # [ mesh_xs, mesh_data ] = gridnd2mesh(g, data)
    #
    # h = patch(isosurface(mesh_xs{:}, mesh_data, level))
    # isonormals(mesh_xs{:}, mesh_data, h)
    # h.FaceColor = color
    # h.EdgeColor = 'none'
    #
    # if applyLight:
    #   lighting phong
    #   camlight left
    #   camlight right
    #
    # view(3)
    #
    # ## 4D Visualization
    # function h = visSetIm4D(g, data, color, level, sliceDim, applyLight)
    # # h = visSetIm4D(g, data, color, level, sliceDim, applyLight)
    # # Visualizes a 4D reachable set
    # #
    # # Takes 6 slices in the dimension sliceDim and shows the 3D projections
    #
    # N = 6
    # spC = 3
    # spR = 2
    # h = cell(N,1)
    # for i = 1:N
    #   subplot(spR, spC, i)
    #   xs = g.min(sliceDim) + i/(N+1) * (g.max(sliceDim) - g.min(sliceDim))
    #
    #   dim = zeros(1, 4)
    #   dim(sliceDim) = 1
    #   [g3D, data3D] = proj(g, data, dim, xs)
    #
    #   # Visualize 3D slices
    #   h{i} = visSetIm3D(g3D, data3D, color, level, applyLight)

    return h
