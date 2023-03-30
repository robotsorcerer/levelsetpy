__all__ = ['upwindFirstFirst']

import copy
import logging
import cupy as cp
import numpy as np
from LevelSetPy.Utilities import *
logger = logging.getLogger(__name__)

def upwindFirstFirst(grid, data, dim, generateAll=False):
    """
     upwindFirstFirst: first order upwind approx of first derivative.

       [ derivL, derivR ] = upwindFirstFirst(grid, data, dim, generateAll)

     Computes a first order directional approximation to the first derivative.

     The generateAll option is used for debugging, and possibly by
       higher order weighting schemes.  Under normal circumstances
       the default (generateAll = false) should be used.

     In fact, since there is only one first order approximation in each
       direction, this argument is completely ignored by this particular function.

     parameters:
       grid	Grid structure (see processGrid.m for details).
       data        Data array.
       dim         Which dimension to compute derivative on.
       generateAll Ignored by this function (optional, default = 0).

       derivL      Left approximation of first derivative (same size as data).
       derivR      Right approximation of first derivative (same size as data).

     Copyright 2004 Ian M. Mitchell (mitchell@cs.ubc.ca).
     This software is used, copied and distributed under the licensing
       agreement contained in the file LICENSE in the top directory of
       the distribution.

     Ian Mitchell, 5/12/03
     modified to avoid permuting, Ian Mitchell, 5/27/03
     modified into upwind form, Ian Mitchell, 1/22/04

     Lekan Molu, 8/21/2021
         Added cupy impl on Nov 18, 21
    """
    if isinstance(data, cp.ndarray):
      data = cp.asarray(data)

    if((dim < 0) or (dim > grid.dim)):
        ValueError('Illegal dim parameter')

    dxInv = 1 / grid.dx.item(dim)

    # How big is the stencil?
    stencil = 1

    # Add ghost cells.
    gdata = grid.bdry[dim](data, dim, stencil, grid.bdryData[dim])

    # Create cell array with array indices.
    sizeData = size(gdata)
    indices1 = []
    for i in range(grid.dim):
        indices1[i] = cp.arange(sizeData[i], dtype=cp.intp)
    indices2 = copy.copy(indices1)

    #Where does the actual data lie in the dimension of interest?
    indices1[dim] = cp.arange(size(gdata, dim), dtype=cp.intp)
    indices2[dim] = indices1[dim] - 1

    #This array includes one extra entry in dimension of interest.
    deriv = dxInv*(gdata[cp.ix_(indices1)] - gdata[cp.ix_(indices2)])

    #Take leftmost grid.N(dim) entries for left approximation.
    indices1[dim] = cp.arange(size(deriv, dim) - 1, dtype=cp.intp)
    derivL = deriv[cp.ix_(indices1)]

    #Take rightmost grid.N(dim) entries for right approximation.
    indices1[dim] = cp.arange(1,size(deriv, dim), dtype=cp.intp)
    derivR = deriv[cp.ix_(indices1)]

    return  derivL, derivR
