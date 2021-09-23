from Utilities import *
import sys, copy
from .check_eq_approx import checkEquivalentApprox
from .ENO3aHelper import upwindFirstENO3aHelper

def upwindFirstENO3a(grid, data, dim, generateAll=False):
    """
     upwindFirstENO3a: third order upwind approx of first deriv by divided diffs.

       [ derivL, derivR ] = upwindFirstENO3a(grid, data, dim, generateAll)

     Computes a third order directional approximation to the first
       derivative, using an Essentially Non-Oscillatory (ENO) approximation.

     The approximation is constructed by a divided difference table,
       which is more efficient (although a little more complicated)
       than using the direct equations from O&F section 3.4
       (see upwindFirstENO3b for that version).

     Details of this scheme can be found in O&F, section 3.3,
       where this scheme is equivalent to including the Q_1, Q_2 and Q_3
       terms of the ENO approximation.

     The generateAll option is used for debugging, and possibly by
       higher order weighting schemes.  Under normal circumstances
       the default (generateAll = false) should be used.

     parameters:
       grid	Grid structure (see processGrid.m for details).
       data        Data array.
       dim         Which dimension to compute derivative on.
       generateAll Return all possible third order upwind approximations.
                     If this boolean is true, then derivL and derivR will
                     be cell vectors containing all the approximations
                     instead of just the ENO approximation.
                     (optional, default = 0)

       derivL      Left approximation of first derivative (same size as data).
       derivR      Right approximation of first derivative (same size as data).

     Copyright 2004 Ian M. Mitchell (mitchell@cs.ubc.ca).
     This software is used, copied and distributed under the licensing
       agreement contained in the file LICENSE in the top directory of
       the distribution.

     Ian Mitchell, 1/23/04
     Lekan on August 16, 2021
    """
    if((dim < 0) or (dim > grid.dim)):
        error('Illegal dim parameter')

    # Check that approximations that should be equivalent are equivalent
    #   (for debugging purposes, only used if generateAll == 1).
    checkEquivalentApproximations = True
    small = 100 * sys.float_info.epsilon             # a small number for "equivalence"

    if(generateAll):
        # We only need the three ENO approximations
        #   (plus the fourth if we want to check for equivalence).
        dL, dR = upwindFirstENO3aHelper(grid, data, dim, checkEquivalentApproximations)

        if(checkEquivalentApproximations):
            # Only the LLL and RRR approximations are not equivalent to at least one
            #   other approximations, so we have several checks.

            # Check corresponding left and right approximations against one another.
            checkEquivalentApprox(dL[1], dR[0], small)
            checkEquivalentApprox(dL[2], dR[1], small)

            # Check the middle approximations.
            checkEquivalentApprox(dL[1], dL[3], small)
            checkEquivalentApprox(dR[1], dR[3], small)
        #---------------------------------------------------------------------------
        # Caller requested all approximations in each direction.
        #   If we requested all four approximations above, strip off the last one.
        derivL = dL[:3]
        derivR = dR[:3]
    else:
        # We need the three ENO approximations
        #   plus the (stripped) divided differences to pick the least oscillatory.
        dL, dR, DD = upwindFirstENO3aHelper(grid, data, dim, False, True)

        #---------------------------------------------------------------------------
        # Create cell array with array indices.
        sizeData = size(data)
        indices1 = []
        for i in range(grid.dim):
            indices1.append(np.arange(sizeData[i], dtype=np.intp))
        indices2 = copy.copy(indices1)

        #---------------------------------------------------------------------------
        # Need to figure out which approximation has the least oscillation.
        #   Note that L and R in this section refer to neighboring divided
        #   difference entries, not to left and right approximations.

        # Pick out minimum modulus neighboring D2 term.
        D2abs = np.abs(DD.D2, order=ORDER_TYPE)
        indices1[dim] = np.arange(size(D2abs, dim)-1, dtype=np.intp)
        indices2[dim] = indices1[dim] + 1
        # print(f'indices1[dim]: {len(indices1[dim])} indices2[dim]: {len(indices2[dim])}')
        # print(D2abs.shape)
        # print(f'D2abs[np.ix_(*indices1): {D2abs[np.ix_(*indices1).shape}, {D2abs[np.ix_(*indices2)].shape}')
        smallerL = (D2abs[np.ix_(*indices1)] < D2abs[np.ix_(*indices2)])
        smallerR = np.logical_not(smallerL)

        #---------------------------------------------------------------------------
        # Figure out smallest modulus D3 terms,
        #   given choice of smallest modulus D2 terms above.
        D3abs = np.abs(DD.D3, order=ORDER_TYPE)
        indices1[dim] = index_array(1,size(D3abs, dim)-1)
        indices2[dim] = indices1[dim] + 1
        smallerTemp = (D3abs[np.ix_(*indices1)] < D3abs[np.ix_(*indices2)])

        indices1[dim] = index_array(1,size(smallerTemp, dim)-1)
        indices2[dim] = indices1[dim] +1
        smallerLL = np.logical_and(smallerTemp[np.ix_(*indices1)], smallerL)
        smallerRL = np.logical_and(smallerTemp[np.ix_(*indices2)], smallerR)
        smallerTemp = np.logical_not(smallerTemp)
        smallerLR = np.logical_and(smallerTemp[np.ix_(*indices1)], smallerL)
        smallerRR = np.logical_and(smallerTemp[np.ix_(*indices2)], smallerR)

        smallerM = np.logical_or(smallerRL, smallerLR)

        #---------------------------------------------------------------------------
        # Pick out the best third order approximation
        indices1[dim] = index_array(1,size(smallerLL, dim)-1)
        derivL = (dL[0] * smallerLL[np.ix_(*indices1)] + dL[1] * smallerM[np.ix_(*indices1)] + dL[2] * smallerRR[np.ix_(*indices1)])

        indices1[dim] = index_array(2,size(smallerLL, dim))
        derivR = (dR[0] * smallerLL[np.ix_(*indices1)]
                + dR[1] * smallerM[np.ix_(*indices1)]
                + dR[2] * smallerRR[np.ix_(*indices1)])

    return derivL, derivR
