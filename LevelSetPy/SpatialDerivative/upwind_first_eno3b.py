__all__ = ['upwindFirstENO3b']

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

from .ENO3bHelper import upwindFirstENO3bHelper
from .check_eq_approx import checkEquivalentApprox

def  upwindFirstENO3b(grid, data, dim, generateAll=0):
    """
     upwindFirstENO3b: third order upwind approx of first deriv by direct calc.

       [ derivL, derivR ] = upwindFirstENO3b(grid, data, dim, generateAll)

     Computes a third order directional approximation to the first
       derivative, using an Essentially Non-Oscillatory (ENO) approximation.

     The approximation is constructed by the equations in O&F, section 3.4
       equations (3.25) - (3.27).  This is an
       alternative to the more efficient divided difference
       table for computing the ENO approximations, which is used in
       upwindFirstENO3a.  In particular, the left and right approximations
       are computed independently in this version.

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
                     instead of just the ENO approximation.  Note that
                     the ordering of these approximations may not be
                     consistent between upwindFirstENO1 and upwindFirstENO2.
                     (optional, default = 0)

       derivL      Left approximation of first derivative (same size as data).
       derivR      Right approximation of first derivative (same size as data).

    Lekan on August 16, 2021
    Added cupy impl on Nov 18, 21
    """
    if isinstance(data, np.ndarray):
      data = np.asarray(data)

    if((dim < 0) or (dim > grid.dim)):
        ValueError('Illegal dim parameter')

    # How big is the stencil?
    stencil = 3

    # Check that approximations that should be equivalent are equivalent
    #   (for debugging purposes, only used if generateAll == 1).
    checkEquivalentApproximations = 1
    small = 100 * eps            # a small number for "equivalence"

    # Add ghost cells.
    gdata = grid.bdry[dim](data, dim, stencil, grid.bdryData[dim])
    if(generateAll):
        # Compute the left and right approximations.
        # No need to build WENO approximation, just return all the ENO approx.
        res = upwindFirstENO3bHelper(grid, gdata, dim, -1)
        derivL = res.eno_approx
        res = upwindFirstENO3bHelper(grid, gdata, dim, +1)
        derivR = res.eno_approx

        #---------------------------------------------------------------------------
        # If necessary, check equivalence of ENO terms.
        # Using notation of (3.25) - (3.27) for phi^i and the Left/Right
        #   choices at the D^1, D^2 and D^3 levels in that order:
        #   For left approximation, phi^1 is LLL, phi^2 is LLR/LRL and phi^3 is LRR;
        #   For right approximation, phi^1 is RRR, phi^2 is RLR/RRL and phi^3 is RLL.
        # Hence the equivalences listed below.
        if(checkEquivalentApproximations):
            checkEquivalentApprox(derivL[1], derivR[2], small)
            checkEquivalentApprox(derivL[2], derivR[1], small)
    else:
        #Compute the left and right ENO approximations.
        res = upwindFirstENO3bHelper(grid, gdata, dim, -1)
        dL, smoothL = res.eno_approx, res.smooth_est

        res = upwindFirstENO3bHelper(grid, gdata, dim, +1)
        dR, smoothR = res.eno_approx, res.smooth_est

        #The best ENO approximant has the smallest smoothness estimate
        derivL = choose(dL, smoothL)
        derivR = choose(dR, smoothR)

    return derivL, derivR



def choose(d, s):
    # deriv = choose(d, s);
    #
    # Helper function to choose least oscillatory ENO approximant.

    choose1over2 = (s[0] < s[1])
    choose1over3 = (s[0] < s[2])
    choose2over3 = (s[1] < s[2])

    deriv = ((choose1over2 and choose1over3) * d[0] \
            + (np.logical_not(choose1over2) and choose2over3) * d[1] \
            + (np.logical_not(choose1over3) and np.logical_not(choose2over3)) * d[2])

    return deriv
