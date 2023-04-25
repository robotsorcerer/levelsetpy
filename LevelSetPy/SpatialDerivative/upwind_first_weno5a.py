__all__ = ['upwindFirstWENO5a']

__author__ 		= "Lekan Molu"
__copyright__ 	= "2021, Hamilton-Jacobi Analysis in Python"
__credits__  	= "There are None."
__license__ 	= "Molux Licence"
__maintainer__ 	= "Lekan Molu"
__email__ 		= "patlekno@icloud.com"
__status__ 		= "Completed"


__author__ 		= "Lekan Molu"
__copyright__ 	= "2021, Hamilton-Jacobi Analysis in Python"
__credits__  	= "There are None."
__license__ 	= "Molux Licence"
__maintainer__ 	= "Lekan Molu"
__email__ 		= "patlekno@icloud.com"
__status__ 		= "Completed"


import copy
import logging
import numpy as np
from LevelSetPy.Utilities import *
logger = logging.getLogger(__name__)


from .ENO3aHelper import upwindFirstENO3aHelper

def upwindFirstWENO5a(grid, data, dim, generateAll=False):
    """
     upwindFirstWENO5a: fifth order upwind approx of first deriv by divided diffs.

       [ derivL, derivR ] = upwindFirstWENO5a(grid, data, dim, generateAll)

     Computes a fifth order directional approximation to the first derivative,
       using a Weighted Essentially Non-Oscillatory (WENO) approximation.

     The ENO approximations are constructed by a divided difference table,
       which is more efficient (although a little more complicated)
       than using the direct equations from O&F section 3.4
       (see upwindFirstWENO5b for that version).

     The smoothness estimates are computed from the first divided difference
       table.  The left and right estimates are computed together,
       taking advantage of the symmetries in the equations.  The results
       should be the same as (3.32) - (3.34) in section 3.4 of O&F.

     The generateAll option is used for debugging, and possibly by
       higher order weighting schemes.  Under normal circumstances
       the default (generateAll = false) should be used.  Notice that
       the generateAll option will just return the three ENO approximations.

     parameters:
       grid	Grid structure (see processGrid.m for details).
       data        Data array.
       dim         Which dimension to compute derivative on.
       generateAll Return all possible third order upwind approximations.
                     If this boolean is true, then derivL and derivR will
                     be cell vectors containing all the approximations
                     instead of just the WENO approximation.  Note that
                     the ordering of these approximations may not be
                     consistent between upwindFirstWENO1 and upwindFirstWENO2.
                     (optional, default = 0)

       derivL      Left approximation of first derivative (same size as data).
       derivR      Right approximation of first derivative (same size as data).

     Copyright Lekan Molu, 8/21/2021
    """
    if((dim < 0) or (dim > grid.dim)):
        error('Illegal dim parameter')

    # How would you like to calculate epsilon?
    #   'constant'         Use constant 1e-6.
    #   'maxOverGrid'      Scale by maximum D1**2 term over the entire grid.
    #   'maxOverNeighbors' Scale by maximum D1**2 term among five neighbors.
    # Compared to 'constant' (fastest), 'maxOverGrid' is about ~3# slower
    #   and 'maxOverNeighbors' about ~17# slower.
    # 'maxOverNeighbors' is the recommended method in O&F equation (3.38).
    epsilonCalculationMethod = 'constant'
    epsilonCalculationMethod = 'maxOverGrid'
    #epsilonCalculationMethod = 'maxOverNeighbors'

    #---------------------------------------------------------------------------
    if(generateAll):
        # We only need the three ENO approximations
        derivL, derivR, _ = upwindFirstENO3aHelper(grid, data, dim, False)
    else:

        # We need the three ENO approximations plus the (unstripped) divided differences to pick the least oscillatory.
        dL, dR, DD = upwindFirstENO3aHelper(grid, data, dim, False, False)

        # For now, use only the first order divided differences.
        D1 = DD.D1

        # Create cell array with array indices.
        sizeData = size(data)
        indices1 = []
        for i in range(grid.dim):
            indices1.append(np.arange(sizeData[i], dtype=np.intp))
        indices2 = copy.copy(indices1)

        terms = 5
        indices = [indices1 for i in range(terms)]

        # print(f"D1: {D1.shape}")
        # create the substencils
        for i in range(terms):
            indices[i][dim] = np.arange(i, size(D1, dim) + i - terms + 1, dtype=np.intp)
            # print(f"indices[{i}][{dim}]: {indices[i][dim]}, min: {indices[i][dim].min()}, max: {indices[i][dim].max()}")
            
        # Smoothness estimates of stencils (see eq. 35 in my paper).
        smooth = cell(3,1)
        smooth[0] = ((13/12) * (D1[np.ix_(*indices[0])] \
                        - 2 * D1[np.ix_(*indices[1])] + D1[np.ix_(*indices[2])]) **2 \
                   + (1/4) * (D1[np.ix_(*indices[0])] - 4 * D1[np.ix_(*indices[1])] \
                              + 3 * D1[np.ix_(*indices[2])]) **2)
        smooth[1] = ((13/12) * (D1[np.ix_(*indices[1])] \
                              - 2 * D1[np.ix_(*indices[2])] \
                              + D1[np.ix_(*indices[3])]) **2 \
                   + (1/4) * (D1[np.ix_(*indices[1])] \
                              - D1[np.ix_(*indices[3])]) **2)
        smooth[2] = ((13/12) * (D1[np.ix_(*indices[2])] \
                              - 2 * D1[np.ix_(*indices[3])] \
                              + D1[np.ix_(*indices[4])]) **2 \
                   + (1/4) * (3 * D1[np.ix_(*indices[2])] \
                              - 4 * D1[np.ix_(*indices[3])] \
                              + D1[np.ix_(*indices[4])]) ** 2)
        
        # Left smoothness estimates just use the left looking portion of
        #   these estimates.  The ENO approximations are in the same order
        #   as in O&F, so we can use the same alpha weights as (3.35) - (3.37).
        smoothL = cell(len(smooth))
        indices1[dim] = np.arange(size(data, dim),dtype=np.intp)

        for i in range(len(smooth)):
            print(f"smooth[{i}]: {smooth[i].shape}")
            print("indices1 \n", indices1)
            smoothL[i] = smooth[i][np.ix_(*indices1)]

        weightL = [ 0.1, 0.6, 0.3 ]

        # Right smoothness estimates are the same, but with D1 in the opposite order.
        #   Fortunately, the estimates are symmetric if we swap v1 for v5,
        #   v2 for v4, and take the right looking portion of the estimates.
        # Note that the ENO approximations (and smoothness estimates)
        #   are in the opposite order as O&F, so we need to reorder the alpha
        #   weights from (3.35) - (3.37).
        smoothR = cell(len(smooth))
        indices2[dim] = np.arange(1,size(data, dim) + 1, dtype=np.intp)
        for i in range(len(smooth)):
            smoothR[i] = smooth[i][np.ix_(*indices2)]

        weightR = [ 0.3, 0.6, 0.1 ]

        #---------------------------------------------------------------------------
        if strcmp(epsilonCalculationMethod, 'constant'):
            epsilonL = 1e-6
            epsilonR = epsilonL
        elif strcmp(epsilonCalculationMethod, 'maxOverGrid'):
            D1squared = D1**2
            epsilonL = 1e-6 * np.max(D1squared.flatten()) + 1e-99
            epsilonR = epsilonL
        elif strcmp(epsilonCalculationMethod, 'maxOverNeighbors'):
            # Implements (3.38) in O&F for computing epsilon.
            D1squared = D1**2
            epsilon = D1squared[np.ix_(*indices[0])]
            for i in range(1, len(indices)):
                epsilon = np.max(epsilon, D1squared[np.ix_(*indices[i])].flatten())
            epsilon = 1e-6 * epsilon + 1e-99
            epsilonL = epsilon[np.ix_(*indices1)]
            epsilonR = epsilon[np.ix_(*indices2)]
        else:
            error(f'Unknown epsilonCalculationMethod  {epsilonCalculationMethod}')

        #---------------------------------------------------------------------------
        # Compute and apply weights to generate a higher order WENO approximation.
        derivL = weightWENO(dL, smoothL, weightL, epsilonL)
        derivR = weightWENO(dR, smoothR, weightR, epsilonR)

    return derivL, derivR


def  weightWENO(d, s, w, epsilon):
    """
     deriv = weightWENO(d, s, w, epsilon)

     Helper function to compute and apply WENO weighting terms.

     See (3.39 - 3.41)  Osher and Fedkiw

     Lekan, Aug 21, 2021
    """

    # Compute weighting terms
    alpha1 = w[0] / (s[0] + epsilon)**2
    alpha2 = w[1] / (s[1] + epsilon)**2
    alpha3 = w[2] / (s[2] + epsilon)**2
    denominator = (alpha1 + alpha2 + alpha3)

    deriv = (alpha1 * d[0] + alpha2 * d[1] + alpha3 * d[2]) / denominator

    return deriv
