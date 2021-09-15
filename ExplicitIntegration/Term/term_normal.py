from Utilities import *

def termNormal(t, y, schemeData):
    """
     termNormal: motion in the normal direction in an HJ PDE with upwinding.

     [ ydot, stepBound, schemeData ] = termNormal(t, y, schemeData)

     Computes an approximation of motion of the interface at speed a(x,t) in
       the normal direction.  The PDE is:

                D_t \phi + a(x,t) \| \grad \phi \| = 0.

       Based on methods outlined in O&F, chapter 6.  The Godunov scheme from
       chapter 6.2 is used.

     parameters:
       t            Time at beginning of timestep.
       y            Data array in vector form.
       schemeData	 A structure (see below).

       ydot	 Change in the data array, in vector form.
       stepBound	 CFL bound on timestep for stability.
       schemeData   The same as the input argument (unmodified).

     schemeData is a structure containing data specific to this type of
       term approximation.  For this function it contains the field(s)

       .grid	 Grid structure (see processGrid.m for details).
       .derivFunc   Function handle to upwinded finite difference
                      derivative approximation.
       .speed	 A description of the normal speed (see below).
       .passVLS     An optional boolean used for vector level sets (see below).
                      Default is 0 (ie ignore vector level sets).

     It may contain additional fields at the user's discretion.

     schemeData.speed can provide the speed in one of two ways:
       1) For time invariant speed, a scalar or an array the same
          size as data.
       2) For general speed, a function handle to a function with prototype
          a = scalarGridFunc(t, data, schemeData), where the output a is the
          scalar/array from (1) and the input arguments are the same as those
          of this function (except that data = y has been reshaped to its
          original size).  In this case, it may be useful to include additional
          fields in schemeData.

     For evolving vector level sets, y may be a cell vector.  If y is a cell
       vector, schemeData may be a cell vector of equal length.  In this case
       all the elements of y (and schemeData if necessary) are ignored except
       the first.  As a consequence, if schemeData.speed is a function handle
       the call to scalarGridFunc will be performed with a regular data array
       and a single schemeData structure (as if no vector level set was present).

     This default behavior of ignoring the vector level set in the call
       to scalarGridFunc may be overridden by setting schemeData.passVLS = 1.
       In this case the data argument (and schemeData argument, if necessary)
       in the call to velocityFunc will be the full cell vectors.  The current
       data array (and schemeData structure, if necessary) will be the first
       element of these cell vectors.  In order to properly reshape the other
       elements of y, the corresponding schemeData structures must contain
       an appropriate grid structure.

     In the notation of OF text,

       data = y	  \phi, reshaped to vector form.
       derivFunc	  Function to calculate phi_i^+-.
       speed	  a.

       delta = ydot  -a \| \grad \phi \|, with upwinded approx to \grad \phi
                       and reshaped to vector form.


     Copyright 2005 Ian M. Mitchell (mitchell@cs.ubc.ca).
     This software is used, copied and distributed under the licensing
       agreement contained in the file LICENSE in the top directory of
       the distribution.

     Ian Mitchell 3/1/04.
     Updated to handle vector level sets.  Ian Mitchell 11/23/04.
     Updated to include passVLS option.  Ian Mitchell 02/14/05.

    Lekan Molu, 08/21/21
    """
    if(iscell(schemeData)):
        thisSchemeData = schemeData[0]
    else:
        thisSchemeData = schemeData

    assert isfield(thisSchemeData, 'grid'), "grid not in schemeData"
    assert isfield(thisSchemeData, 'derivFunc'), "derivFunc not in schemeData"
    assert isfield(thisSchemeData, 'speed'), "speed not in schemeData"

    grid = thisSchemeData.grid

    #For most cases, we are interested in the first implicit surface function.
    if(iscell(y)):
        data = y[0].reshape(grid.shape, order='F')
    else:
        data = y.reshape(grid.shape, order='F')

    #Get speed field.
    if(isfloat(thisSchemeData.forcing)):
        forcing = thisSchemeData.forcing

    elif(callable(thisSchemeData.forcing)):

        if(iscell(y)):
            if(isfield(thisSchemeData, 'passVLS') and thisSchemeData.passVLS):
                #Pass the vector level set information through.
                numY = len(y)
                data = cell(numY, 1)
                for i in range(numY):
                    if(iscell(schemeData)):
                        data[i] = y[i].reshape(schemeData[i].grid.shape, order='F')
                    else:
                        data[i] = y[i].reshape(schemeData.grid.shape, order='F')

                speed = thisSchemeData.speed(t, dataV, schemeData)

            else:
                #Ignore any vector level set.
                speed = thisSchemeData.speed(t, data, thisSchemeData)

        else:
            # There is no vector level set.
            speed = thisSchemeData.speed(t, data, thisSchemeData)
    else:
        error('schemeData.speed must be a scalar, array or function handle')

    # In the end, all we care about is the magnitude of the gradient.
    magnitude = zeros(size(data))

    # In this case, keep track of stepBound for each node until the very
    #   end (since we need to divide by the appropriate gradient magnitude).
    stepBoundInv = zeros(size(data))

    # Determine the upwind direction dimension by dimension
    for i in range(grid.dim):
        # Get upwinded derivative approximations.
        derivL, derivR = thisSchemeData.derivFunc(grid, data, i)

        # Effective velocity in this dimension (scaled by \|\grad \phi\|).
        prodL = speed * derivL
        prodR = speed * derivR
        magL = np.abs(prodL)
        magR = np.abs(prodR)

        # Determine the upwind direction.
        #   Either both sides agree in sign (take direction in which they agree),
        #   or characteristics are converging (take larger magnitude direction).
        flowL = ((prodL >= 0) and (prodR >= 0)) or ((prodL >= 0) and (prodR <= 0) and (magL >= magR))
        flowR = ((prodL <= 0) and (prodR <= 0)) or ((prodL >= 0) and (prodR <= 0) and (magL < magR))

        # For diverging characteristics, take gradient = 0
        #   (so we don't actually need to calculate this term).
        #flow0 = ((prodL <= 0) & (prodR >= 0))

        # Now we know the upwind direction, add its contribution to \|\grad \phi\|.
        magnitude += (derivL**2 * flowL + derivR**2 * flowR)

        # CFL condition: sum of effective velocities from O&F (6.2).
        effectiveVelocity = (magL * flowL + magR * flowR)
        dxInv = 1 / grid.dx[i]
        stepBoundInv += (dxInv@effectiveVelocity)

    # Finally, calculate speed * \|\grad \phi\|
    magnitude = np.sqrt(magnitude)
    delta = speed * magnitude

    # Find the most restrictive timestep bound.
    nonZero = np.nonzero(magnitude > 0)
    stepBoundInvNonZero = stepBoundInv[nonZero] / magnitude[nonZero]
    stepBound = 1 / np.max(stepBoundInvNonZero)

    # Reshape output into vector format and negate for RHS of ODE.
    ydot = expand(-delta.flatten(order='F'), 1)

    return ydot, stepBound, schemeData
