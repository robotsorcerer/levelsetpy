__all__ = ["artificialDissipationLLF"]

__author__ 		= "Lekan Molu"
__copyright__ 	= "2021, Hamilton-Jacobi Analysis in Python"
__credits__  	= "There are None."
__license__ 	= "MIT License"
__maintainer__ 	= "Lekan Molu"
__email__ 		= "patlekno@icloud.com"
__status__ 		= "Completed"


import copy
import cupy as cp
import numpy as np
from LevelSetPy.Utilities import *

def artificialDissipationLLF(t, data, derivL, derivR, schemeData):
    """
     artificialDissipationLLF: local Lax-Friedrichs dissipation calculation.

     [ diss, stepBound ] = ...
                     artificialDissipationLLF(t, data, derivL, derivR, schemeData)

     Calculates the stabilizing dissipation for the local Lax-Friedrichs
       numerical Hamiltonian, which is known to be monotone.  The method is
       "local" because it optimizes alpha = |\partial H(x,p) / \partial p_i| at
       each node x over the entire range of p for the entire grid except that
       the range of p_i is restricted to the range for that node (ie p_i^+ to
       p_i^- at that node).  The result is guaranteed to generate the same or
       less dissipation than regular (global) Lax-Friedrichs.

     Based on methods outlined in O&F, chapter 5.3.1 (the LLF scheme).

     Parameters:
       t            Time at beginning of timestep.
       data         Data array.
       derivL	 Cell vector with left derivatives of the data.
       derivR	 Cell vector with right derivatives of the data.
       schemeData	 A structure (see below).

       diss	 Local Lax-Friedrichs dissipation for each node.
       stepBound	 CFL bound on timestep for stability.

     schemeData is a structure containing data specific to this type of
       term approximation.  For this function it contains the field(s)

       .grid	 Grid structure.
       .partialFunc Function handle to extrema of \partial H(x,p) / \partial p.


     schemeData.partialFunc should have prototype

             alpha = partialFunc(t, data, derivMin, derivMax, schemeData, dim)

       where t and schemeData are passed directly from this function, data = y
       has been reshaped into its original size, dim is the dimension of
       interest, and derivMin and derivMax are both cell vectors (of length
       grid.dim) containing the elements of the minimum and maximum costate p =
       \grad \phi (respectively).  The range of nodes over which this minimum
       and maximum is taken depends on the choice of dissipation function.  The
       return value should be an array (the size of data) containing alpha_dim:

        maximum_{p \in [ derivMin, derivMax ] | \partial H(x,p) / \partial p_dim |


     in the notation of Osher and Fedkiw text,
       data	  \phi.
       derivL	  \phi_i^- (all dimensions i are in the cell vector).
       derivR	  \phi_i^+ (all dimensions i are in the cell vector).
       partialFunc	  \alpha^i (dimension i is an argument to partialFunc).
       diss	  all the terms in \hat H except the H term.


     Reference: Osher, S., & Shu, C.-W. (1991). High-Order Essentially Nonoscillatory
                Schemes for Hamilton-Jacobi Equations. Society for Industrial and
                Applied Mathematics, 28(4), 907–922. https://doi.org/10.2514/1.9320

     Ian Mitchell 2/13/04
     Lekan Molu Aug 21, 2021
    """
    assert isfield(schemeData, 'grid'), "grid not in schemeData"
    assert isfield(schemeData, 'partialFunc'), "partialFunc not in schemeData"

    grid = schemeData.grid

    #First find the global LF range of costate, since we use that in all
    #the other dimensions.
    derivMin = cell(grid.dim)
    derivMax = cell(grid.dim)
    derivDiff = cell(grid.dim)

    for i in range(grid.dim):
        # Get derivative bounds over entire grid (scalars).
        derivMinL = cp.min(derivL[i].flatten())
        derivMinR = cp.min(derivR[i].flatten())
        derivMin[i] = min(derivMinL, derivMinR)

        derivMaxL = cp.max(derivL[i].flatten())
        derivMaxR = cp.max(derivR[i].flatten())
        derivMax[i] = max(derivMaxL, derivMaxR)

        # Get derivative differences at each node.
        derivDiff[i] = derivR[i] - derivL[i]

    # We need copies of the costate range.
    derivMinCopy = copy.copy(derivMin)
    derivMaxCopy = copy.copy(derivMax)

    #---------------------------------------------------------------------------
    # Now calculate the dissipation.  Since alpha is the effective speed of
    #   the flow, it provides the CFL timestep bound too.
    diss = 0
    stepBoundInv = 0
    for i in range(grid.dim):

        # For each dimension, LLF restricts the range of that dimension's
        #   costate at each node to the range between left and right
        #   approximations at that node.
        derivMin[i] = cp.minimum(derivL[i], derivR[i])
        derivMax[i] = cp.maximum(derivL[i], derivR[i])

        alpha = schemeData.partialFunc(t, data, derivMin, derivMax, schemeData, i)

        # Restore full range of costate.
        derivMin[i] = derivMinCopy[i]
        derivMax[i] = derivMaxCopy[i]

        diss += (0.5 * derivDiff[i] * alpha)
        stepBoundInv +=  (alpha / grid.dx.item(i))

    stepBound = (1 / stepBoundInv).get().item()

    return  diss, stepBound
