__all__ = ["shapeRectangleByCorners"]

import logging
import numpy as np
from LevelSetPy.Utilities.matlab_utils import *

logger = logging.getLogger(__name__)

def shapeRectangleByCorners(grid, lower=None, upper=None):
    """
     shapeRectangleByCorners: implicit surface function for a (hyper)rectangle.

       data = shapeRectangleByCorners(grid, lower, upper)

     Creates an implicit surface function (close to signed distance)
       for a coordinate axis aligned (hyper)rectangle specified by its
       lower and upper corners.

     Can be used to create intervals, slabs and other unbounded shapes
       by choosing components of the corners as +-Inf.

     The default parameters for shapeRectangleByCenter and
       shapeRectangleByCorners produce different rectangles.

     Inp.t Parameters:

       grid: Grid structure (see processGrid.m for details).

       lower: Vector specifying the lower corner.  May be a scalar, in
       which case the scalar is multiplied by a vector of ones of the
       appropriate length.  Defaults to 0.

       upper: Vector specifying the upper corner.  May be a scalar, in which
       case the scalar is multiplied by a vector of ones of the appropriate
       length.  Defaults to 1.  Note that all(lower < upper) must hold,
       otherwise the implicit surface will be empty.

     Output Parameters:

       data: Output data array (of size grid.size) containing the implicit
       surface function.

      Lekan Molu, September, 2021
    """
    #Default parameter values.
    if not np.any(lower):
        lower = zeros(grid.dim, 1)
    elif(numel(lower) == 1):
        lower *= ones(grid.dim, 1)

    if not np.any(upper):
        upper = ones(grid.dim, 1)
    elif(numel(upper) == 1):
        upper *= ones(grid.dim, 1)

    #Implicit surface function calculation.
    #This is basically the intersection (by max operator) of halfspaces.
    #While each halfspace is generated by a signed distance function,
    #   the resulting intersection is not quite a signed distance function.
    data = np.maximum(grid.xs[0] - upper[0], lower[0] - grid.xs[0])
    for i in range(1, grid.dim):
        data = np.maximum(data, grid.xs[i] - upper[i])
        data = np.maximum(data, lower[i] - grid.xs[i])

    #---------------------------------------------------------------------------
    # Warn the user if there is no sign change on the grid
    #  (ie there will be no implicit surface to visualize).
    if(np.all(data.flatten() < 0) or (np.all(data.flatten() > 0))):
        logger.warn(f'Implicit surface not visible because function has '
                'single sign on grid')

    return data
