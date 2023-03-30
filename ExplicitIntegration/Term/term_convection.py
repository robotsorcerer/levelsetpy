__all__ = ["termConvection"]

import copy
import numpy as np
from LevelSetPy.Utilities import *

def termConvection(t, y, schemeData):
    """
     termConvection: approximate a convective term in an HJ PDE with upwinding.

     [ ydot, stepBound, schemeData ] = termConvection(t, y, schemeData)

     Computes an approximation of motion by a constant velocity field V(x,t)
     for a Hamilton-Jacobi PDE (often called convective or advective flow).
     The PDE is:

                D_t \phi = -V(x,t) \dot \grad \phi.

     Based on methods outlined in O&F, chapter 3.  The more conservative CFL
     condition (3.10) is used.

     Inp.t Parameters:

       t: Time at beginning of timestep.

       y: Data array in vector form.

       schemeData: A structure (see below).

     Output Parameters:

       ydot: Change in the data array, in vector form.

       stepBound: CFL bound on timestep for stability.

       schemeData: The same as the inp.t argument (unmodified).

     schemeData is a structure containing data specific to this type of term
     approximation.  For this function it contains the field(s):

       .grid: Grid structure (see processGrid.m for details).

       .derivFunc: Function handle to upwinded finite difference derivative
       approximation.

       .velocity: A description of the velocity field (see below).

       .passVLS: An optional boolean used for vector level sets (see below).
       Default is 0 (ie ignore vector level sets).

     It may contain additional fields at the user's discretion.

     schemeData.velocity can describe the velocity field in one of two ways:

       1) For time invariant velocity fields, a cell vector (length grid.dim)
          of flow velocities, where each cell contains a scalar or an array
          the same size as data.

       2) For general velocity fields, a function handle to a function
          with prototype velocity = velocityFunc(t, data, schemeData),
          where the output velocity is the cell vector from (1) and
          the inp.t arguments are the same as those of this function
          (except that data = y has been reshaped to its original size).
          In this case, it may be useful to include additional fields in
          schemeData.

     For evolving vector level sets, y may be a cell vector.  If y is a cell
     vector, schemeData may be a cell vector of equal length.  In this case all
     the elements of y (and schemeData if necessary) are ignored except the
     first.  As a consequence, if schemeData.velocity is a function handle the
     call to velocityFunc will be performed with a regular data array and a
     single schemeData structure (as if no vector level set was present).

     This default behavior of ignoring the vector level set in the call to
     velocityFunc may be overridden by setting schemeData.passVLS = 1.  In this
     case the data argument (and schemeData argument, if necessary) in the call
     to velocityFunc will be the full cell vectors.  The current data array
     (and schemeData structure, if necessary) will be the first element of
     these cell vectors.  In order to properly reshape the other elements of y,
     the corresponding schemeData structures must contain an appropriate grid
     structure.

     In the notation of OF text,

       data = y	  \phi, reshaped to vector form.
       derivFunc	  Function to calculate phi_i^+-.
       velocity	  V(x).

       delta = ydot  -V \dot \grad \phi, with upwinded approx to \grad \phi
                       and reshaped to vector form.

    Lekan Molu, 08/21/2021
    """
    #For vector level sets, ignore all the other elements.
    if(iscell(schemeData)):
        thisSchemeData = schemeData[0]
    else:
        thisSchemeData = schemeData

    assert isfield(thisSchemeData, 'grid'), "grid not in schemeData"
    assert isfield(thisSchemeData, 'velocity'), "velocity not in schemeData"
    assert isfield(thisSchemeData, 'derivFunc'), "derivFunc not in schemeData"

    grid = thisSchemeData.grid

    if(iscell(y)):
        data = y[0].reshape(grid.shape)
    else:
        data = y.reshape(grid.shape)

    # Get velocity field.
    if(iscell(thisSchemeData.velocity)):
        velocity = thisSchemeData.velocity

    elif(callable(thisSchemeData.velocity)):

        if(iscell(y)):
            if(isfield(thisSchemeData, 'passVLS') and thisSchemeData.passVLS):
                #Pass the vector level set information through.
                numY = len(y)
                vectorData = cell(numY)
                for i in range(numY):
                    if(iscell(schemeData)):
                        vectorData[i] = y[i].reshape(schemeData[i].grid.shape)
                    else:
                        vectorData[i] = y[i].reshape(schemeData.grid.shape)
                velocity = thisSchemeData.velocity(t, vectorData, schemeData)

            else:
                # Ignore any vector level set.
                velocity = thisSchemeData.velocity(t, data, thisSchemeData)

        else:
            # There is no vector level set.
            velocity = thisSchemeData.velocity(t, data, thisSchemeData)

    else:
        error('schemeData.velocity must be a cell vector or a function handle')


    # Approximate the convective term dimension by dimension.
    delta = zeros(size(data))
    stepBoundInv = 0
    for i in range(grid.dim):
        #Get upwinded derivative approximations.
        derivL, derivR = thisSchemeData.derivFunc(grid, data, i)

    # Figure out upwind direction.
    v = velocity[i]
    flowL = (v < 0)
    flowR = (v > 0)

    # Approximate convective term with upwinded derivatives
    # (where v == 0 derivative doesn't matter).
    deriv = derivL * flowR + derivR * flowL

    # Dot product requires sum over dimensions.
    delta += deriv * v

    # CFL condition.  Note that this is conservative we really should do
    # the summation over the entire grid and then take the maximum,
    # rather than maximizing for each dimension and then summing.
    stepBoundInv += np.max(np.abs(v)) / grid.dx[i]

    stepBound = 1 / stepBoundInv

    # Reshape output into vector format and negate for RHS of ODE.
    ydot = expand(-delta.flatten(), 1)

    return ydot, stepBound, schemeData
