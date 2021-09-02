from Utilities import *
from .hessian import hessianSecond

def curvatureSecond(grid, data):
    """
     curvatureSecond: second order centered difference approx of the curvature.

       [ curvature, gradMag ] = curvatureSecond(grid, data)

     Computes a second order centered difference approximation to the curvature.

           \kappa = divergence(\grad \phi / | \grad \phi |)

     See O&F section 1.4 for more details.  In particular, this routine
       implements equation 1.8 for calculating \kappa.

     parameters:
       grid	Grid structure (see processGrid.m for details).
       data        Data array.

       curvature   Curvature approximation (same size as data).
       gradMag	Magnitude of gradient |\grad \phi|
                     Incidentally calculated while finding curvature,
                     also second order centered difference.

     Copyright 2004 Ian M. Mitchell (mitchell@cs.ubc.ca).
     This software is used, copied and distributed under the licensing
       agreement contained in the file LICENSE in the top directory of
       the distribution.

     Ian Mitchell, 6/3/03
     Lekan Molu, 08/21/2021
    """

    # Get the first and second derivative terms.
    second, first  = hessianSecond(grid, data)

    # Compute gradient magnitude.
    gradMag2 = first[0]**2
    for i in range(1,grid.dim):
        gradMag2 += first[i]**2

    gradMag = np.sqrt(gradMag2)

    curvature = zeros(size(data))
    for i in range(grid.dim):
        curvature += second[i,i] * (gradMag2 - first[i]**2)
        for j in range(i - 1):
            curvature -=(2 * first[i] * first[j] * second[i,j])

    # Be careful not to stir the wrath of "Divide by Zero".
    #  Note that gradMag == 0 implies curvature == 0 already, since all the
    #  terms in the curvature approximation involve at least one first dervative.
    nonzero = np.nonzero(gradMag > 0)
    curvature[nonzero] = curvature[nonzero] / gradMag[nonzero]**3

    return curvature, gradMag
