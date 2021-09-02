from Utilities import *
from .ENO3bHelper import upwindFirstENO3bHelper
from .upwind_first_weno5a import weightWENO

def upwindFirstWENO5b(grid, data, dim, generateAll=0):
    """
     upwindFirstWENO5b: fifth order upwind approx of first deriv by direct calc.

       [ derivL, derivR ] = upwindFirstWENO5b(grid, data, dim, generateAll)

     Computes a fifth order directional approximation to the first derivative,
       using a Weighted Essentially Non-Oscillatory (WENO) approximation.

     The approximation is constructed by the equations in O&F, section 3.4
       equations (3.25) - (3.40).  In particular, the three ENO
       approximations are computed by (3.25) - (3.27). This is an
       alternative to the more efficient divided difference
       table for computing the ENO approximations, which is used in
       upwindFirstWENO5a.  In particular, the left and right approximations
       are computed independently in this version.

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

     Copyright 2004 Ian M. Mitchell (mitchell@cs.ubc.ca).
     This software is used, copied and distributed under the licensing
       agreement contained in the file LICENSE in the top directory of
       the distribution.

     Copyright Lekan Molu, 8/21/2021 Adopted from
             2004 Ian M. Mitchell (mitchell@cs.ubc.ca)
             LevelSets Toolbox. Ian Mitchell, 1/26/04
    """

    if((dim < 0) or (dim > grid.dim)):
        error('Illegal dim parameter')

    # How big is the stencil?
    stencil = 3

    # Add ghost cells.
    gdata = grid.bdry[dim](data, dim, stencil, grid.bdryData[dim])


    #---------------------------------------------------------------------------
    if(generateAll):
        # Compute the left and right approximations.
        # No need to build WENO approximation, just return all the ENO approx.
        derivL = upwindFirstENO3bHelper(grid, gdata, dim, -1)
        derivR = upwindFirstENO3bHelper(grid, gdata, dim, +1)
    else:
        #Compute the left and right ENO approximations.
        dL, smoothL, epsilonL = upwindFirstENO3bHelper(grid, gdata, dim, -1)
        dR, smoothR, epsilonR = upwindFirstENO3bHelper(grid, gdata, dim, +1)

        w = [ 0.1, 0.6, 0.3 ]
        #Compute and apply weights to generate a higher order WENO approximation.
        derivL = weightWENO(dL, smoothL, w, epsilonL)
        derivR = weightWENO(dR, smoothR, w, epsilonR)

    return derivL, derivR
