__all__ = ["shapeCylinder"]

__author__ 		= "Lekan Molu"
__copyright__ 	= "2021, Hamilton-Jacobi Analysis in Python"
__credits__  	= "There are None."
__license__ 	= "MIT License"
__maintainer__ 	= "Lekan Molu"
__email__ 		= "patlekno@icloud.com"
__status__ 		= "Completed"

import numpy as np
from .utils import check_target
from levelsetpy.utilities.matlab_utils import *
from .utils import check_target

def shapeCylinder(grid, axis_align=[], center=None, radius=1):
    """
     shapeCylinder: implicit surface function for a cylinder.

       data = shapeCylinder(grid, axis_align, center, radius)

     Creates an implicit surface function (actually signed distance) for a
       coordinate axis aligned cylinder whose axis runs parallel to the
       coordinate dimensions specified in axis_align.

     Can be used to create:
       Intervals, circles and spheres (if axis_align is empty).
       Slabs (if axis_align contains all dimensions except one).

     Parameters:
     ==========

       grid: Grid structure (see processGrid.py for details).

       axis_align: Vector specifying indices of coordinate axes with which the
       cylinder is aligned.  Defaults to the empty vector (eg: the cylinder is
       actually a sphere).

       center: Vector specifying a point at the center of the cylinder.
       Entries in the ignored dimensions are ignored.  May be a scalar, in
       which case the scalar is multiplied by a vector of ones of the
       appropriate length.  Defaults to 0 (eg centered at the origin).

       radius: Scalar specifying the radius of the cylinder.  Defaults to 1.

     Output:
     =======
       data: Output data array (of size grid.size) containing the implicit
       surface function.

     Translated August 2, 2021 | Lekan Molu
    ---------------------------------------------------------------------------
     Default parameter values.
    """

    if not np.any(center):
        center = zeros(grid.dim, 1)
    elif(numel(center) == 1):
        center = center * ones(grid.dim, 1, dtype=np.float64)

    #---------------------------------------------------------------------------
    # Signed distance function calculation.
    data = np.zeros((grid.shape))
    for i in range(grid.dim):
        if(i != axis_align):
            data += (grid.xs[i] - center[i])**2
    data = np.sqrt(data) - radius

    #---------------------------------------------------------------------------
    # Warn the user if there is no sign change on the grid
    #  (ie there will be no implicit surface to visualize).
    check_target(data)
    return data
