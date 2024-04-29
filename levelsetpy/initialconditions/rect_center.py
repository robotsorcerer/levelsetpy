__all__ = ["shapeRectangleByCenter"]

__author__ 		= "Lekan Molu"
__copyright__ 	= "2021, Hamilton-Jacobi Analysis in Python"
__credits__  	= "There are None."
__license__ 	= "MIT License"
__maintainer__ 	= "Lekan Molu"
__email__ 		= "patlekno@icloud.com"
__status__ 		= "Completed"

import numpy as np
from levelsetpy.utilities.matlab_utils import *
from .rect_corners import shapeRectangleByCorners

def shapeRectangleByCenter(grid, center=None, widths=None):
    """
     shapeRectangleByCenter: implicit surface function for a (hyper)rectangle.

       data = shapeRectangleByCenter(grid, center, widths)

     Creates an implicit surface function (close to signed distance) for a
     coordinate axis aligned (hyper)rectangle specified by its center and
     widths in each dimension.

     Can be used to create intervals and slabs by choosing components of the
     widths as +Inf.

     The default parameters for shapeRectangleByCenter and
     shapeRectangleByCorners produce different rectangles.

     Inp.t Parameters:

       grid: Grid structure (see processGrid.m for details).

       center: Vector specifying center of rectangle.  May be a scalar, in
       which case the scalar is multiplied by a vector of ones of the
       appropriate length.  Defaults to 0 (eg centered at the origin).

       widths: Vector specifying (full) widths of each side of the rectangle.
       May be a scalar, in which case all dimensions have the same width.
       Defaults to 1.

     Output Parameters:

       data: Output data array (of size grid.size) containing the implicit
       surface function.

     Lekan Molu, Sept 07, 2021
    """
    #Default parameter values.
    if not np.any(center):
        center = zeros(grid.dim, 1)
    elif(numel(center) == 1):
        center *= ones(grid.dim, 1)

    if not np.any(widths):
        widths = ones(grid.dim, 1);
    elif numel(widths) == 1:
        widths *= ones(grid.dim, 1)

    #---------------------------------------------------------------------------
    # Implicit surface function calculation.
    #   This is basically the intersection (by max operator) of halfspaces.
    #   While each halfspace is generated by a signed distance function,
    #   the resulting intersection is not quite a signed distance function.

    # For the computation, we really want the lower and upper corners.
    lower = zeros(grid.dim, 1);
    upper = zeros(grid.dim, 1);
    for i in range(grid.dim):
        lower[i] = center[i] - 0.5 * widths[i]
        upper[i] = center[i] + 0.5 * widths[i]

    data = shapeRectangleByCorners(grid, lower, upper);

    #---------------------------------------------------------------------------
    # Warn the user if there is no sign change on the grid
    #  (ie there will be no implicit surface to visualize).
    if(np.all(data.flatten() < 0) or (np.all(data.flatten() > 0))):
        logger.warn(f'Implicit surface not visible because function has '
                'single sign on grid')

    return data