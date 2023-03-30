__all__ = ["termReinit"]

import copy
import numpy as np
from LevelSetPy.Utilities import *

def termReinit(t, y, schemeData):
    """
     termReinit: a Godunov solver for the reinitialization HJ PDE.

     [ ydot, stepBound, schemeData ] = termReinit(t, y, schemeData)

     Computes a Godunov approximation to motion by the reinitialization
     equation.  While the reinitialization equation is a general nonlinear HJ
     PDE, such a Godunov approximation is the least dissipative monotone
     approximation (less dissipative than Roe-Fix or Lax-Friedrichs).  The
     reinitialization equation is

                D_t \phi = -sign(\phi_0)(\|\grad \phi\| - 1).

     where phi_0 is the initial conditions.  Solving the reinitialization
     equation turns an implicit surface function into a signed distance
     function.  It is iterative, and often slower than a fast marching method
     however, it can use high order approximations and can start directly from
     the implicit surface function without needing to explicitly find the
     implicit surface (although the subcell fix discussed below does in some
     sense find the surface).

     The term and concept of reinitialization was originally proposed by Chopp
     in 1993, but the reinitialization PDE used here comes from

       M. Sussman, P. Smereka & S. Osher, "A Level Set Method for Computing
       Solutions to Incompressible Two Phase Flow," J. Computational Physics,
       v. 119, pp. 145-159 (1994).

     It is discussed in O&F chapter 7.4.  The translation between the
     parameters of this function and the notation in O&F,

       data = y	  \phi, reshaped to vector form.
       derivFunc	  Function to calculate phi_i^+-.
       initial	  \phi_0
       delta = ydot  -S(\phi_0)(|\grad \phi| - 1)

     The Gudonov solver used below comes from from appendix A.3 of

       R. Fedkiw, T. Aslam, B. Merriman & S. Osher, "A Non-Oscillatory
       Eulerian Approach to Interfaces in Multimaterial Flows (the Ghost
       Fluid Method)," J. Computational Physics, v. 152, pp. 457-492 (1999).

     The subcell fix option is implemented based on

       Giovanni Russo & Peter Smereka, "A Remark on Computing Distance
       Functions," J. Computational Physics, v. 163, pp. 51-67 (2000),
       doi:10.1006/jeph.2000.6553

     Note that Russo & Smereka is based on the schemes from Sussman, Smereka &
     Osher and Sussman & Fatemi (1999), which may not be identical to Fedkiw,
     Aslam, Merriman & Osher.

     The smoothed sgn() approximation is given in the subfunction smearedSign.
     Note that if the subcell fix is applied, the smoothed sgn() function is
     not used (although a similar smoothing is implicitly applied).

     Inp.t Parameters:

       t: Time at beginning of timestep.

       y: Data array in vector form.

       schemeData:	A structure (see below).

     Output Parameters:

       ydot: Change in the data array, in vector form.

       stepBound: CFL bound on timestep for stability.

       schemeData: The same as the inp.t argument (unmodified).

     The parameter schemeData is a structure containing data specific to this
     type of term approximation.  For this function it contains the field(s)

       .grid: Grid structure (see processGrid.m for details).

       .derivFunc: Function handle to upwinded finite difference derivative
       approximation.

       .initial: Array, same size as data.  Initial implicit surface function
       (used to determine on which side of surface node should lie).
       Consequently, on the first reinitialization timestep, this array should
       be equal to data.

       .subcell_fix_order: Integer.  Specifies whether to apply the subcell fix
       from Russo & Smereka to nodes near the interface, and if the fix is
       applied specifies what order of accuracy to use.  Specify order 0 to
       turn off the fix.  At present, only orders 0 and 1 are supported.
       Optional.  Default = 1.

     The schemeData structure may contain addition fields at the user's
     discretion.

     For evolving vector level sets, y may be a cell vector.  If y is a cell
     vector, schemeData may be a cell vector of equal length.  In this case all
     the elements of y (and schemeData if necessary) are ignored except the
     first.

    Lekan Molu, 08/21/21
    """

    #The subcell fix has some options.

    # Use the robust signed distance function (17) or the simple one (13)?
    # The simple one often fails due to divide by zero errors, so be careful.
    robust_subcell = 1

    # Small positive parameter that appears in the robust version.  In
    # fact, we will use this as a relative value with respect to grid.dx
    robust_small_epsilon = 1e6 * eps

    #For vector level sets, ignore all the other elements.
    if iscell(schemeData):
        thisSchemeData = schemeData[0]
    else:
        thisSchemeData = schemeData

    # Check for required fields.
    assert isfield(thisSchemeData, 'grid'), "grid not in schemeData"
    assert isfield(thisSchemeData, 'derivFunc'), "derivFunc not in schemeData"
    assert isfield(thisSchemeData, 'initial'), "initial not in schemeData"

    grid = thisSchemeData.grid

    if iscell(y):
        data = y[0].reshape(grid.shape, order='F')
    else:
        data = y.reshape(grid.shape, order='F')

    if isfield(thisSchemeData, 'subcell_fix_order'):
        if thisSchemeData.subcell_fix_order==0:
            apply_subcell_fix = 0
        elif thisSchemeData.subcell_fix_order== 1:
            apply_subcell_fix = 1
            subcell_fix_order = 1

        else:
            error(f'Reinit subcell fix order of accuracy {thisSchemeData.subcell_fix_order} not supported')
    else:
        #Default behavior is to apply the simplest subcell fix.
        apply_subcell_fix = 1
        subcell_fix_order = 1

    if apply_subcell_fix:
        # The sign function is only used far from the interface, so we do
        # not need to smooth it.
        S = np.sign(thisSchemeData.initial)
    else:
        # Smearing factor for the smooth approximation of the sign function.
        dx = np.max(grid.dx)
        sgnFactor = dx**2

        # Sign function (smeared) identifies on which side of surface each node
        # lies.
        S = smearedSign(grid, thisSchemeData.initial, sgnFactor)

    """
    Compute Godunov derivative approximation for each dimension.  This
    code is used for the PDE far from the interface, or for all nodes if
    the subcell fix is not applied.
    """
    deriv = cell(grid.dim, 1)

    for i in range(grid.dim):
        derivL, derivR = thisSchemeData.derivFunc(grid, data, i)

        # For Gudunov's method, check characteristic directions
        #   according to left and right derivative approximations.

        # Both directions agree that flow is to the left.
        flowL = ((S * derivR <= 0) and (S * derivL <= 0))

        # Both directions agree that flow is to the right.
        flowR = ((S * derivR >= 0) and (S * derivL >= 0))

        # Diverging flow entropy condition requires choosing deriv = 0
        #   (so we don't actually have to calculate this term).
        #flow0 = ((S * derivR >  0) and (S * derivL <  0))

        # Converging flow, need to check which direction arrives first.
        flows = ((S * derivR <  0) and (S * derivL >  0))
        if(np.any(flows.flatten())):
            conv = np.where(flows)
            s = zeros(size(flows))
            s[conv] *=(np.abs(derivR[conv]) - np.abs(derivL[conv]))/(derivR[conv] - derivL[conv])

            # If s == 0, both directions arrive at the same time.
            #   Assuming continuity, both will produce same result, so pick one.
            flowL[conv] = flowL[conv] or (s[conv] < 0)
            flowR[conv] = flowR[conv] or (s[conv] >= 0)

        deriv[i] = derivL * flowR + derivR * flowL

    # Compute magnitude of gradient.
    mag = zeros(size(grid.xs[0]))
    for i in range(grid.dim):
        mag += deriv[i]**2
    mag = np.max(np.sqrt(mag), eps)

    # Start with constant term in the reinitialization equation.
    delta = -S

    # Compute change in function and bound on step size.
    stepBoundInv = 0
    for i in range(grid.dim):
        # Effective velocity field (for timestep bounding).
        v = S * deriv[i]/mag

        # Update just like a velocity field.
        delta += v * deriv[i]

        # CFL condition using effective velocity.
        stepBoundInv = stepBoundInv + np.max(np.abs(v[:])) / grid.dx[i]

    if apply_subcell_fix:

        if subcell_fix_order==1:
            # Most of the effort below -- specifically computation of the distance to
            # the interface D -- depends only on thisSchemeData.initial, so
            # recomputation could be avoided if there were some easy way to
            # memoize the results between timesteps.  It could be done by
            # modifying schemeData, but that has a rather high overhead and could
            # lead to bugs if the user fiddles with schemeData.  So for now, we
            # recompute at each timestep.

            # Set up some index cell vectors.  No ghost cells will be used, since
            # nodes near the edge of the computational domain should not be
            # near the interface.  Where necessary, we will modify the stencil
            # near the edge of the domain.
            indexL = cell(grid.dim, 1)
            for d in range(grid.dim):
                indexL[d] = quickarray(0, grid.N[d])
            indexR = indexL

            # Compute denominator in (13) or (16) or (23).  Note that we
            # have moved the delta x term into this denominator to treat
            # the case when delta x is not the same in each dimension.
            denom = zeros(size(data))
            for d in range(grid.dim):
                dx_inv = 1 / grid.dx[d]

                # Long difference used in (13) and (23).  For the nodes near the
                # edge of the computational domain, we will just use short differences.
                indexL[d] = [0] + quickarray(0, grid.N[d] - 1)
                indexR[d] = [1]+ quickarray(1, grid.N[d])
                diff2 = (0.5 * dx_inv@(thisSchemeData.initial[indexR[:]] - thisSchemeData.initial[indexL[:]])) ** 2

                if robust_subcell:
                    # Need the short differences.
                    indexL[d] = quickarray(0,grid.N[d] - 1)
                    indexR[d] = quickarray(1,grid.N[d])
                    short_diff2 = (dx_inv@(thisSchemeData.initial[indexR[:]] - thisSchemeData.initial[indexL[:]])) ** 2

                    # All the various terms of (17).
                    diff2[indexL[:]] = np.max(diff2[indexL[:]], short_diff2)
                    diff2[indexR[:]] = np.max(diff2[indexR[:]], short_diff2)
                    diff2 = np.max(diff2, robust_small_epsilon ** 2)

                # Include this dimension's contribution to the distance.
                denom += diff2

                # Reset the index vectors.
                indexL[d] = quickarray(0, grid.N[d])
                indexR[d] = quickarray(0, grid.N[d])

            denom = np.sqrt(denom)

            # Complete (13) or (16) or (23).  Note that delta x was already included
            # in the denominator calculation above, so it does not appear.
            D = thisSchemeData.initial / denom

            # We do need to know which nodes are near the interface.
            near = isNearInterface(thisSchemeData.initial)

            # Adjust the update.  The delta x that appears in (15) or (22)
            # comes from the smoothing in (14), so we choose the maximum
            # delta x in this case (guarantees sufficient smoothing no
            # matter what the direction of the interface).  For grids with
            # different delta x, this choice may require more
            # reinitialization steps to achieve desired results.
            delta = (delta * (not near)+(S * np.abs(data) - D) / np.max(grid.dx) * near)

            # We will not adjust the CFL step bound.  By Russo & Smereka, the
            # adjusted update has a bound of 1, and the regular scheme above should
            # already have that same upper bound.

        else:
            error(f'Reinit subcell fix order of accuracy {subcell_fix_order} not supported')

    stepBound = 1 / stepBoundInv

    # Reshape output into vector format and negate for RHS of ODE.
    ydot = expand(-delta[:], 1)

    return ydot, stepBound, schemeData

def smearedSign(grid, data, sgnFactor):
    """
    s = smearedSign(grid, data)

    Helper function to generated a smeared signum function.

    This version (with sgnFactor = dx**2) is (7.5) in O&F chapter 7.4.
    """
    s = data / np.sqrt(data**2 + sgnFactor)

    return s
