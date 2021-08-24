from utils import *

def termLaxFriedrichs(t, y, schemeData):
    """
      termLaxFriedrichs: approximate H(x,p) term in an HJ PDE with Lax-Friedrichs.

        [ ydot, stepBound, schemeData ] = termLaxFriedrichs(t, y, schemeData)

      Computes a Lax-Friedrichs (LF) approximation of a general Hamilton-Jacobi
      equation.  Global LF, Local LF, Local Local LF and Stencil LF are
      implemented by choosing different dissipation functions.  The PDE is:

                 D_t \phi = -H(x, t, \phi, D_x \phi).

      Based on methods outlined in O&F, chapter 5.3 and 5.3.1.

      Input parameters:

        t: Time at beginning of timestep.

        y: Data array in vector form.

        schemeData: A structure (see below).

      Output parameters:

        ydot: Change in the data array, in vector form.

        stepBound: CFL bound on timestep for stability.

        schemeData: The input structure, possibly modified.


      schemeData is a structure containing data specific to this type of
      term approximation.  For this function it contains the field(s):

        .grid: Grid structure (see processGrid.m for details).

        .derivFunc: Function handle to upwinded finite difference
        derivative approximation.

        .dissFunc: Function handle to LF dissipation calculator.

        .hamFunc: Function handle to analytic hamiltonian H(x,p).

        .partialFunc: Function handle to extrema of \partial H(x,p) / \partial p.

      Note that options for derivFunc and dissFunc are provided as part of the
      level set toolbox, while hamFunc and partialFunc depend on the exact term
      H(x,p) and are user supplied.  Note also that schemeData may contain
      addition fields at the user's discretion for example, fields containing
      parameters useful to hamFunc or partialFunc.


      schemeData.hamFunc should have one of the prototypes:

              [ hamValue ]             = hamFunc(t, data, deriv, schemeData)

              [ hamValue, schemeData ] = hamFunc(t, data, deriv, schemeData)

      where t and schemeData are passed directly from this function, data = y
      has been reshaped into its original size, and deriv is a cell vector (of
      length grid.dim) containing the elements of the costate p = \grad \phi.
      The return value should be an array (the size of data) containing H(x,p).
      Optionally, a modified schemeData structure may also be returned. Any
      modifications to the schemeData structure will be visible immediately to
      schemeData.partialFunc on this timestep.  NOTE: Matlab's nargout()
      function is used to determine the number of output arguments.  In some
      cases nargout() returns -1 for example, if hamFunc is an anonymous
      function or an object method.  In those cases the second prototype (with
      two output parameters) MUST BE USED.


      For details on schemeData.partialFunc, see the dissipation functions.


      For evolving vector level sets, y may be a cell vector.  If y is a cell
      vector, schemeData may be a cell vector of equal length.  In this case
      all the elements of y (and schemeData if necessary) are ignored except
      the first.  As a consequence, calls to schemeData.hamFunc and
      schemeData.partialFunc will be performed with a regular data array and a
      single schemeData structure (as if no vector level set was present).


      In the notation of OF text:

        data	  \phi.
        derivFunc	  Function to calculate phi_i^+-.
        dissFunc      Function to calculate the terms with alpha in them.
        hamFunc	  Function to calculate analytic H.
        partialFunc	  \alpha^i (dimension i is an argument to partialFunc).

        update	  -\hat H.


      Copyright 2004 Ian M. Mitchell (mitchell@cs.ubc.ca).
      This software is used, copied and distributed under the licensing
        agreement contained in the file LICENSE in the top directory of
        the distribution.

      Ian Mitchell 5/13/03
      Calling parameters significantly modified, Ian Mitchell 2/11/04.
      Updated to handle vector level sets.  Ian Mitchell 11/23/04.

      Lekan Aug 18, 2021
    """
    #---------------------------------------------------------------------------
    # For vector level sets, ignore all the other elements.
    if(iscell(schemeData)):
        thisSchemeData = schemeData[0]
    else:
        thisSchemeData = schemeData

    assert isfield(thisSchemeData, 'grid'),  'grid not in struct thisschemeData'
    assert isfield(thisSchemeData, 'derivFunc'),  'derivFunc not in struct thisschemeData'
    assert isfield(thisSchemeData,'dissFunc'),  'dissFunc not in struct thisschemeData'
    assert isfield(thisSchemeData,'hamFunc'), 'hamFunc not in struct thisschemeData'
    assert isfield(thisSchemeData,'partialFunc'),  'partialFunc not in struct thisschemeData'

    grid = thisSchemeData.grid

    #---------------------------------------------------------------------------
    if(iscell(y)):
        data = y[0].reshape(grid.shape)
    else:
        data = y.reshape(grid.shape)

    #---------------------------------------------------------------------------
    # Get upwinded and centered derivative approximations.
    derivL = cell(grid.dim, 1)
    derivR = cell(grid.dim, 1)
    derivC = cell(grid.dim, 1)

    for i in range(grid.dim):
        derivL[i], derivR[i] = thisSchemeData.derivFunc(grid, data, i)
        derivC[i] = 0.5 * (derivL[i] + derivR[i])

    ham, thisSchemeData = thisSchemeData.hamFunc(t, data, derivC, thisSchemeData)
    # Need to store the modified schemeData structure.
    if(iscell(schemeData)):
        schemeData[0] = thisSchemeData
    else:
        schemeData = thisSchemeData

    # Lax-Friedrichs dissipative stabilization.
    diss, stepBound = thisSchemeData.dissFunc(t, data, derivL, derivR, thisSchemeData)

    # Calculate update: (unstable) analytic hamiltonian
    #                   - (dissipative) stabiliziation.
    delta = ham - diss

    #---------------------------------------------------------------------------
    # Reshape output into vector format and negate for RHS of ODE.
    ydot = -delta.flatten()

    return ydot, stepBound, schemeData
