__all__ = ["artificialDissipationGLF"]

__author__ 		= "Lekan Molu"
__copyright__ 	= "2021, Hamilton-Jacobi Analysis in Python"
__credits__  	= "There are None."
__license__ 	= "Molux Licence"
__maintainer__ 	= "Lekan Molu"
__email__ 		= "patlekno@icloud.com"
__status__ 		= "Completed"


import numpy as np
from LevelSetPy.Utilities import *

def artificialDissipationGLF(t, data, derivL, derivR, schemeData):
    """
     artificialDissipationGLF: (global) Lax-Friedrichs dissipation calculation.

     diss, stepBound =artificialDissipationGLF(t, data, derivL, derivR, schemeData)

     Calculates the stabilizing dissipation for the (global) Lax-Friedrichs
       numerical Hamiltonian, which is known to be monotone.  The method is
       "global" because it optimizes alpha = |\partial H(x,p) / \partial p| at
       each node x over the entire range of p for the entire grid.  "Local"
       schemes restrict the range of nodes over which we look for extrema in p
       (and hence restrict the extrema in alpha).

     Based on methods outlined in Osher and Fedkiw, chapter 5.3.1 (the first LF scheme).

     Parameters:
       t            Time at beginning of timestep.
       data         Data array.
       derivL	 Cell vector with left derivatives of the data.
       derivR	 Cell vector with right derivatives of the data.
       schemeData	 A structure (see below).

       diss	 Global Lax-Friedrichs dissipation for each node.
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

     Lekan Molux, November 18, 2021
    """

    #---------------------------------------------------------------------------
    if not isfield(schemeData, 'grid'):
        raise ValueError(f'grid is not a structure')
    if not isfield(schemeData, 'partialFunc'):
        raise ValueError(f'partialFunc is not a structure')

    #---------------------------------------------------------------------------
    grid = schemeData.grid

    #---------------------------------------------------------------------------
    # Global LF stability dissipation.
    derivMin = cell(grid.dim)
    derivMax = cell(grid.dim)
    derivDiff = cell(grid.dim)

    # Revusut this
    for i in range(grid.dim):
        # Get derivative bounds over entire grid (scalars).
        derivMinL = np.min(derivL[i].flatten())
        derivMinR = np.min(derivR[i].flatten())
        derivMin[i] = min(derivMinL, derivMinR)

        derivMaxL = np.max(derivL[i].flatten())
        derivMaxR = np.max(derivR[i].flatten())
        derivMax[i] = max(derivMaxL, derivMaxR)

        # Get derivative differences at each node.
        derivDiff[i] = derivR[i] - derivL[i]
    #---------------------------------------------------------------------------
    # Now calculate the dissipation.  Since alpha is the effective speed of
    #   the flow, it provides the CFL timestep bound too.
    diss = 0
    stepBoundInv = 0
    for i in range(grid.dim):
        alpha = schemeData.partialFunc(t, data, derivMin, derivMax, \
                      schemeData, i)

        diss += (0.5 * derivDiff[i] * alpha)
        if isinstance(alpha, np.ndarray):
          #from Osher and Fedkiw, the coeffs are
          # set to the max possible values of |H_{x|y}| respectively
          alpha = np.max(alpha.flatten())


        stepBoundInv += (alpha / grid.dx.item(i))

    stepBound = (1 / stepBoundInv).item()

    return diss, stepBound
