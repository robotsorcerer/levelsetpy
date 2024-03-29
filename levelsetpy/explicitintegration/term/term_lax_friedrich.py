__all__ = ["termLaxFriedrichs"]

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
from levelsetpy.utilities import *

def termLaxFriedrichs(t, y, schemeData):
    """
      termLaxFriedrichs: approximate H(x,p) term in an HJ PDE with Lax-Friedrichs.

        [ ydot, stepBound, schemeData ] = termLaxFriedrichs(t, y, schemeData)

      Computes a Lax-Friedrichs (LF) approximation of a general Hamilton-Jacobi
      equation.  Global LF, Local LF, Local Local LF and Stencil LF are
      implemented by choosing different dissipation functions.  The PDE is:

                 D_t \phi = -H(x, t, \phi, D_x \phi).

      Based on methods outlined in O&F, chapter 5.3 and 5.3.1.
      For evolving vector level sets, y may be a list.  If y is a list,
        schemeData may be a list of equal length.  Here, all elements of y
        (and schemeData if necessary) are ignored except the first.  As a
        consequence, calls to schemeData.hamFunc and schemeData.partialFunc
        will be performed with a regular data array and a single schemeData
        structure (as if no vector level set was present).

        In the notation of O&F text:
          data:	        \phi.
          CoStateCalc:	Function to calculate phi_i^+-.
          dissFunc:     Function to calculate the terms with alpha in them.
          hamFunc:      Function to calculate analytic H.
          partialFunc:	\alpha^i (dimension i is an argument to partialFunc).
          update:	  -\hat H.

      Parameters
      ----------
        t: Time at beginning of timestep.
        y: Data array in vector form.
        schemeData:       A bundle structure withe fields:
            .grid:        Grid structure (see processGrid.py for details).
            .CoStateCalc:   Function handle to upwinded finite difference
                          derivative approximation.
            .dissFunc:    Function handle to LF dissipation calculator.
            .hamFunc:     Function handle to analytic hamiltonian H(x,p).

                          Signature/Prototypes
                          --------------------
                          hamValue = hamFunc(t, data, deriv, schemeData)
                          hamValue, schemeData = hamFunc(t, data, deriv, schemeData)

                          Parameters
                          ----------
                          .t/schemeData,passed directly from this func;
                          .data:= y, reshaped into its original size;
                          .deriv: grid.dim length list containing
                                  costate elements p = \grad \phi.
            .partialFunc: Function handle to extrema of \partial H(x,p) / \partial p.
                          More details? See the dissipation functions.
            Note that options for derivFunc and dissFunc are provided as part of the
            level set toolbox, while hamFunc and partialFunc depend on the exact term
            H(x,p) and are user supplied.  Note also that schemeData may contain
            addition fields at the user's discretion for example, fields containing
            parameters useful to hamFunc or partialFunc.

      Returns
      -------
        ydot: Change in the data array, in vector form.
        stepBound: CFL bound on timestep for stability.
        schemeData: The data structure, possibly modified.

      Author: Lekan Aug 18, 2021
    """
    #---------------------------------------------------------------------------
    # For vector level sets, ignore all the other elements.
    if(iscell(schemeData)):
        thisSchemeData = copy.copy(schemeData[0])
    else:
        thisSchemeData = copy.copy(schemeData)

    assert isfield(thisSchemeData, 'grid'),  'grid not in bundle thisschemeData'
    assert isfield(thisSchemeData, 'CoStateCalc'),  'CoStateCalc not in bundle thisschemeData'
    assert isfield(thisSchemeData, 'dissFunc'),  'dissFunc not in bundle thisschemeData'
    assert isfield(thisSchemeData, 'hamFunc'), 'hamFunc not in bundle thisschemeData'
    assert isfield(thisSchemeData, 'partialFunc'),  'partialFunc not in bundle thisschemeData'

    grid = copy.copy(thisSchemeData.grid)

    if(iscell(y)):
        data = y[0].reshape(grid.shape)
    else:
        data = y.reshape(grid.shape)

    # Get upwinded and centered derivative approximations.
    derivL = [cp.nan for i in range(grid.dim)]
    derivR = [cp.nan for i in range(grid.dim)]
    derivC = [cp.nan for i in range(grid.dim)]

    # Calculate the co-states of value function
    for i in range(grid.dim):
        derivL[i], derivR[i] = thisSchemeData.CoStateCalc(grid, data, i)
        derivC[i] = 0.5 * (derivL[i] + derivR[i])

    # Analytic Hamiltonian with centered difference derivatives.
    result = thisSchemeData.hamFunc(t, data, derivC, thisSchemeData)
    if isinstance(result, tuple):
        ham, thisSchemeData = result
        # Need to store the modified schemeData structure.
        if(iscell(schemeData)):
            schemeData[0] = copy.copy(thisSchemeData)
        else:
            schemeData = copy.copy(thisSchemeData)
    else:
        ham = result

    # Lax-Friedrichs dissipative stabilization.
    diss, stepBound = thisSchemeData.dissFunc(t, data, derivL, derivR, thisSchemeData)
    delta = ham - diss

    #---------------------------------------------------------------------------
    # Reshape output into vector format and negate for RHS of ODE.
    ydot = cp.expand_dims(-delta.flatten(), 1)

    return ydot, stepBound, schemeData
