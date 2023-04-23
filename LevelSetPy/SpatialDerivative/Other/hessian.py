from LevelSetPy.BoundaryCondition import addGhostAllDims
from LevelSetPy.Utilities import *

def hessianSecond(grid, data):
    """
     hessianSecond: second order centered difference approx of the Hessian.

       [ second, first ] = hessianSecond(grid, data)

     Computes a second order centered difference approximation to the Hessian
       (the second order mixed spatial derivative of the data).

     parameters:
       grid	Grid structure (see processGrid.m for details).
       data        Data array.

       second      2D cell array containing centered approx to Hessian's terms.
                     To save space, only lower left half of Hessian is given
                     (since mixed partials are derivative order independent).
                         second{i,j} = d^2 data / dx_i dx_j   if j < i
                                     = d^2 data / dx_i^2      if j = i
                                     = []                     if j > i
       first       1D cell array containing centered approx to gradient
                     (incidentally computed while finding second).
                         first[i]    = d data / dx_i

     Every nonempty element of second (and first) is the same size as
       the data array.

     Why is a gradient approximation provided?
       A gradient approximation is part of the process of computing the mixed
       partial terms in the Hessian, so returning its value requires
       little extra computation.  Note that this gradient is a second order
       centered difference approximation, so it is inappropriate for use in
       the convection term of a PDE (upwinding should be used for such terms).

     Copyright 2004 Ian M. Mitchell (mitchell@cs.ubc.ca).
     This software is used, copied and distributed under the licensing
       agreement contained in the file LICENSE in the top directory of
       the distribution.

     Ian Mitchell, 6/3/03
    """
    dxInv = np.divide(1, grid.dx)

    # How big is the stencil?
    stencil = 1

    # Add ghost cells to every dimension.
    data = addGhostAllDims(grid, data, stencil)

    #
    # We need indices to the real data.
    indReal = cell(grid.dim, 1)
    for i in range(grid.dim):
        indReal[i] = [1 + x for x in quickarray(0, grid.N[i] + stencil)]

    # Also indices to the whole data set (including ghost cells).
    indAll = cell(grid.dim, 1)
    for i in range(grid.dim):
        indAll[i] = quickarray(0, grid.N([i] + 2 * stencil))

    # Centered first partials (gradient approximation).
    first = cell(grid.dim, 1)
    for i in range(grid.dim):
        # leave the ghost cells on other dimensions intact (for mixed partials below)
        indices1 = indAll
        indices2 = indAll
        indices1[i] = indReal[i] + 1
        indices2[i] = indReal[i] - 1
        first[i] = 0.5 * dxInv[i]@(data[indices1[:]] - data(indices2[:]))

    #
    # Centered second partials (Hessian approximation).
    #   We will only fill the lower half of second,
    #   since mixed partials' derivative ordering doesn't matter.
    second = np.empty((grid.dim, grid.dim))
    for i  in range(grid.dim):
        # First, the pure second partials.
        #   Get rid of ghost cells on other dimensions.
        indices1 = indReal
        indices2 = indReal
        indices1[i] = indices1[i] + 1
        indices2[i] = indices2[i] - 1
        second[i,i] = dxInv[i]**2 * (data[indices1[:]] - 2 * data[indReal[:]] + data[indices2[:]])

        # Now the mixed partials.
        for j in range(i - 1):
            # Get rid of ghost cells in dimensions without derivatives.
            indices1 = indReal
            indices2 = indReal
            # In already differentiated dimension, we have no ghost cells.
            indices1[i] = quickarray(0, grid.N[i])
            indices2[i] = quickarray(0, grid.N[i])
            # Now take a centered difference in second direction.
            indices1[j] = indReal[j] + 1
            indices2[j] = indReal[j] - 1

            second[i,j] =  0.5 * dxInv[j] * (first[i][indices1[:]] - first[i](indices2[:]))
    # If the user wants the gradient approximation,
    #   strip unnecessary ghost cells from first partials.
    for i in range(grid.dim):
        indices1 = indReal
        # In already differentiated dimension, we have no ghost cells.
        indices1[i] = quickarray(0, grid.N[i])
        first[i] = first[i][indices1[:]]

    return second, first
