__all__ = ['upwindFirstENO3']

__author__ 		= "Lekan Molu"
__copyright__ 	= "2021, Hamilton-Jacobi Analysis in Python"
__credits__  	= "There are None."
__license__ 	= "MIT License"
__maintainer__ 	= "Lekan Molu"
__email__ 		= "patlekno@icloud.com"
__status__ 		= "Completed"

import copy
import logging
import torch
import numpy as np
from levelsetpy.utilities import *
logger = logging.getLogger(__name__)


from levelsetpy.spatialderivative.check_eq_approx import checkEquivalentApprox
from levelsetpy.spatialderivative.ENO3aHelper import upwindFirstENO3aHelper

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


     Lekan on August 16, 2021

        Added cupy impl on Nov 18, 21
    """
    if isinstance(data, np.ndarray):
      data = torch.as_tensor(data)

    if((dim < 0) or (dim > grid.dim)):
        raise ValueError('Illegal dim parameter')

    # Check that approximations that should be equivalent are equivalent
    #   (for debugging purposes, only used if generateAll == 1).
    checkEquivalentApproximations = True
    small = 100 * eps             # a small number for "equivalence"

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
            indices1.append(torch.arange(sizeData[i], dtype=torch.int64))
        indices2 = copy.copy(indices1)

        #---------------------------------------------------------------------------
        # Need to figure out which approximation has the least oscillation.
        #   Note that L and R in this section refer to neighboring divided
        #   difference entries, not to left and right approximations.

        # Pick out minimum modulus neighboring D2 term.
        D2abs = torch.abs(DD.D2)
        indices1[dim] = torch.arange(size(D2abs, dim)-1, dtype=torch.int64)
        indices2[dim] = indices1[dim] + 1

        smallerL = (D2abs[torch.meshgrid(*indices1, indexing='ij')] < D2abs[torch.meshgrid(*indices2, indexing='ij')])
        smallerR = torch.logical_not(smallerL)

        #---------------------------------------------------------------------------
        # Figure out smallest modulus D3 terms,
        #   given choice of smallest modulus D2 terms above.
        D3abs = torch.abs(DD.D3)
        indices1[dim] = torch.arange(size(D3abs, dim)-1, dtype=torch.int64)
        indices2[dim] = copy.copy(indices1[dim]) + 1
        smallerTemp = (D3abs[torch.meshgrid(*indices1, indexing='ij')] < D3abs[torch.meshgrid(*indices2, indexing='ij')])

        indices1[dim] = torch.arange(size(smallerTemp, dim)-1, dtype=torch.int64)
        indices2[dim] = copy.copy(indices1[dim]) +1
        smallerLL = torch.logical_and(smallerTemp[torch.meshgrid(*indices1, indexing='ij')], smallerL)
        smallerRL = torch.logical_and(smallerTemp[torch.meshgrid(*indices2, indexing='ij')], smallerR)
        smallerTemp = torch.logical_not(smallerTemp)
        smallerLR = torch.logical_and(smallerTemp[torch.meshgrid(*indices1, indexing='ij')], smallerL)
        smallerRR = torch.logical_and(smallerTemp[torch.meshgrid(*indices2, indexing='ij')], smallerR)

        smallerM = torch.logical_or(smallerRL, smallerLR)

        #---------------------------------------------------------------------------
        # Pick out the best third order approximation
        indices1[dim] = torch.arange(size(smallerLL, dim)-1, dtype=torch.int64)
        derivL = (dL[0] * smallerLL[torch.meshgrid(*indices1, indexing='ij')] + \
                  dL[1] * smallerM[torch.meshgrid(*indices1, indexing='ij')] + \
                  dL[2] * smallerRR[torch.meshgrid(*indices1, indexing='ij')])

        indices1[dim] = torch.arange(1,size(smallerLL, dim), dtype=torch.int64)
        derivR = (dR[0] * smallerLL[torch.meshgrid(*indices1, indexing='ij')]
                    + dR[1] * smallerM[torch.meshgrid(*indices1, indexing='ij')]
                    + dR[2] * smallerRR[torch.meshgrid(*indices1, indexing='ij')])

    return derivL, derivR
