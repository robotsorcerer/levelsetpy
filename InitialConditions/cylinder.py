from Utilities import zeros, ones, numel, logger, np

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

     Input Parameters:

       grid: Grid structure (see processGrid.m for details).

       axis_align: Vector specifying indices of coordinate axes with which the
       cylinder is aligned.  Defaults to the empty vector (eg: the cylinder is
       actually a sphere).

       center: Vector specifying a point at the center of the cylinder.
       Entries in the ignored dimensions are ignored.  May be a scalar, in
       which case the scalar is multiplied by a vector of ones of the
       appropriate length.  Defaults to 0 (eg centered at the origin).

       radius: Scalar specifying the radius of the cylinder.  Defaults to 1.

     Output Parameters:

       data: Output data array (of size grid.size) containing the implicit
       surface function.

     Copyright 2004 Ian M. Mitchell (mitchell@cs.ubc.ca).
     This software is used, copied and distributed under the licensing
       agreement contained in the file LICENSE in the top directory of
       the distribution.

     Ian Mitchell, 6/23/04
     $Date: 2011-03-18 16:56:16 -0700 (Fri, 18 Mar 2011) $
     $Id: shapeCylinder.m 60 2011-03-18 23:56:16Z mitchell $

     Translated August 2, 2021 | Lekan Molu
    ---------------------------------------------------------------------------
     Default parameter values.
    """

    if not np.any(center):
        center = zeros(grid.dim, 1);
    elif(numel(center) == 1):
        center = center * ones(grid.dim, 1, dtype=np.float64);

    #---------------------------------------------------------------------------
    # Signed distance function calculation.
    data = np.zeros((grid.shape));
    for i in range(grid.dim):
        if(i != axis_align):
            data += (grid.xs[i] - center[i])**2
    data = np.sqrt(data) - radius;

    #---------------------------------------------------------------------------
    # Warn the user if there is no sign change on the grid
    #  (ie there will be no implicit surface to visualize).
    if(np.all(data.flatten() < 0) or (np.all(data.flatten() > 0))):
        logger.warn(f'Implicit surface not visible because function has '
                'single sign on grid');
    return data
