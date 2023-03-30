

import numpy as np
from LevelSetPy.Utilities import zeros, ones, numel, logger, np
from LevelSetPy.Helper.Math import cellMatrixMultiply, cellMatrixAdd


def shapeHyperplane(grid, normal, point):
    """
     shapeHyperplane: implicit surface function for a hyperplane.

       data = shapeHyperplane(grid, normal, point)

     Creates a signed distance function for a hyperplane.

     Inp.t Parameters:

       grid: Grid structure (see processGrid.m for details).

       normal:  Column vector specifying the outward normal of the hyperplane.

       point: Vector specifying a point through which the hyperplane passes.
       Defaults to the origin.

     Output Parameters:

       data: Output data array (of size grid.size) containing the implicit
       surface function.

     Copyright 2004 Ian M. Mitchell (mitchell@cs.ubc.ca).
     This software is used, copied and distributed under the licensing
       agreement contained in the file LICENSE in the top directory of
       the distribution.

     Ian Mitchell, 6/23/04
     $Date: 2009-09-03 16:34:07 -0700 (Thu, 03 Sep 2009) $
     $Id: shapeHyperplane.m 44 2009-09-03 23:34:07Z mitchell $

     Lekan Molu, September 07, 2021
     """

    if not np.any(point):
        point = zeros(grid.dim, 1)

    #Normalize the normal to be a unit vector.
    normal = normal / np.norm(normal)

    #---------------------------------------------------------------------------
    #Signed distance function calculation. #TODO
    #This operation is just phi = n^T (x - p), but over the entire grid of x.
    data = cellMatrixMultiply(list(normal>T), \
                                cellMatrixAdd(grid.xs, list(-point)))

    # In fact, the cellMatrix operation generates a 1x1 cell matrix
    #   whose contents are the data array.
    data = data[0]

    #---------------------------------------------------------------------------
    # Warn the user if there is no sign change on the grid
    #  (ie there will be no implicit surface to visualize).
    if(np.all(data.flatten() < 0) or (np.all(data.flatten() > 0))):
        logger.warn(f'Implicit surface not visible because function has '
                'single sign on grid')

    return data
