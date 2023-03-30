__all__ = ["shapeHyperplaneByPoints"]


import numpy as np
from LevelSetPy.Utilities import *


def shapeHyperplaneByPoints(grid, points, positivePoint):
    """
     shapeHyperplaneByPoints: implicit surface function for a hyperplane.

       data = shapeHyperplaneByPoints(grid, points, positivePoint)

     Creates a signed distance function for a hyperplane.  Unlike
     shapeHyperplane, this version accepts a list of grid.dim points which lie
     on the hyperplane.

     The direction of the normal (which determines which side of the hyperplane
     has positive values) is determined by one of two methods:

       1) If the parameter positivePoint is provided, then the normal
       direction is chosen so that the value at this point is positive.

       2) If the parameter positivePoint is not provided, then it is assumed
       that the points defining the hyperplane are given in "clockwise" order
       if the normal points out of the "clock".  This method does not work
       in 2D.

     Input Parameters:

       grid: Grid structure (see processGrid.m for details).

       points: Matrix specifying the points through which the hyperplane should
       pass.  Each row is one point.  This matrix must be square of dimension
       grid.dim.

       positivePoint: Vector of length grid.dim specifying a point which lies
       on the positive side of the interface.  This point should be within the
       bounds of the grid.  Optional.  The method for determining the normal
       direction to the hyperplane depends on whether this parameter is
       supplied; see the discussion above for more details.  It is an error if
       this point lies on the hyperplane defined by the other points.

     Output Parameters:

       data: Output data array (of size grid.size) containing the implicit
       surface function for the hyperplane.
    """
    #---------------------------------------------------------------------------
    # For the positivePoint parameter, what is "too close" to the interface?
    small = 1e3 * np.finfo(float).eps

    #---------------------------------------------------------------------------
    if not positivePoint:
        check_positive_point = 0
    else:
        check_positive_point = 1

    #---------------------------------------------------------------------------
    # Check that we have the correct number of points,
    #   and they are linearly independent.
    assert size(points) == (grid.dim, grid.dim), 'Number of points must be equal to grid dimension'

    #---------------------------------------------------------------------------
    # We single out the first point.  Lines from this point to all the others
    # should lie on the hyperplane.
    point0 = points[0,:]
    A = points[1:,:] - np.tile(point0, (grid.dim - 1, 1))

    # Extract the normal to the hyperplane.
    normal = null(A)

    # Check to see that it is well defined.
    if(normal.shape[1] != 1)
        error('There does not exist a unique hyperplane through these points')

    #---------------------------------------------------------------------------
    # Signed distance function calculation.
    #   This operation is just phi = n^T (x - p), but over the entire grid of x.
    data = cellMatrixMultiply(num2cell(normal.T), cellMatrixAdd(grid.xs, num2cell(-point0')))

    # In fact, the cellMatrix operation generates a 1x1 cell matrix
    #   whose contents are the data array.
    data = data[0]

    #---------------------------------------------------------------------------
    # The procedure above generates a correct normal assuming that the data
    # points are given in a clockwise fashion.  If the user supplies
    # parameter positivePoint, we need to use a different test.
    if check_positive_point:
        positivePointCell = num2cell(positivePoint)
        positivePointValue = interpn(grid.xs{:}, data, positivePointCell{:})
    if np.isnan(positivePointValue):
        error('positivePoint must be within the bounds of the grid.')
    elif(abs(positivePointValue) < small):
        error('positivePoint parameter is too close to the hyperplane.')
    elif(positivePointValue < 0):
        data = -data

    #---------------------------------------------------------------------------
    # Warn the user if there is no sign change on the grid
    #  (ie there will be no implicit surface to visualize).
    if(np.all(data.flatten() < 0) or (np.all(data.flatten() > 0))):
    logger.warn(f'Implicit surface not visible because function has '
            'single sign on grid')

    return data
