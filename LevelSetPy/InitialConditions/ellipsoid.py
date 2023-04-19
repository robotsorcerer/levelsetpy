__all__ = ["shapeEllipsoid"]

__author__ 		= "Lekan Molu"
__copyright__ 	= "2021, Hamilton-Jacobi Analysis in Python"
__credits__  	= "There are None."
__license__ 	= "Molux Licence"
__maintainer__ 	= "Lekan Molu"
__email__ 		= "patlekno@icloud.com"
__status__ 		= "Completed"


import warnings
import numpy as np
from .utils import check_target
from LevelSetPy.Utilities.matlab_utils import *
from .utils import check_target

def shapeEllipsoid(grid, center=None, radius=1.0):
    """
     shapeEllipsoid: implicit surface function for an ellipsoid.

       data = shapeEllipsoid(grid, center, radius)

     Creates an implicit surface function (actually signed distance) for
        an ellipsoid.

     Parameters:
     ==========

       grid: Grid structure (see processGrid.py for details).

       center: Vector specifying a point at the center of the ellipsoid.
       Entries in the ignored dimensions are ignored.  May be a scalar, in
       which case the scalar is multiplied by a vector of ones of the
       appropriate length.  Defaults to 0 (eg centered at the origin).

       radius: Scalar specifying the major axis of the ellipsoid.  Defaults to 1.

     Output:
     =======
       data: Output data array (of size grid.size) containing the implicit
       surface function.

     Translated March 28, 2023 | Lekan Molu
    ---------------------------------------------------------------------------
     Default parameter values.
    """
    if not np.any(center):
        center = zeros(grid.dim, 1)
    elif(numel(center) == 1):
        center = center * ones(grid.dim, 1, dtype=np.float64)


    #---------------------------------------------------------------------------
    # Signed distance function calculation.
    data = (grid.xs[0] - center[0])**2
    data += 4.0*(grid.xs[1] - center[1])**2
    if grid.dim==3:
        data += 9.0*(grid.xs[2] - center[2])**2
    data = data - radius

     #---------------------------------------------------------------------------
    # Warn the user if there is no sign change on the grid
    #  (ie there will be no implicit surface to visualize).
    if(np.all(data.flatten() < 0) or (np.all(data.flatten() > 0))):
        warnings.warn(f'Implicit surface not visible because function has '
                'single sign on grid')
    return data
