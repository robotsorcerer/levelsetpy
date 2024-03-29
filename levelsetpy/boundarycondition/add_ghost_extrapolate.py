
__all__ = ["addGhostExtrapolate"]
 
__author__ 		= "Lekan Molu"
__copyright__ = "2021, Hamilton-Jacobi Analysis in Python"
__credits__  	= "There are None."
__license__ 	= "MIT License"
__maintainer__= "Lekan Molu"
__email__ 		= "patlekno@icloud.com"
__status__ 		= "Completed, Circa, August Week I, 2021."
__revised__   = "May 09, 2023"

import copy
import logging
import cupy as cp
import numpy as np
from levelsetpy.utilities import *
logger = logging.getLogger(__name__)

def addGhostExtrapolate(dataIn, dim, width=None, ghostData=None):
    """
     addGhostExtrapolate: add ghost nodes, values extrapolated from bdry nodes.

       dataOut = addGhostExtrapolate(dataIn, dim, width, ghostData)

     Creates ghost nodes to manage the boundary conditions for the array dataIn.

     This script fills the ghost nodes with data linearly extrapolated
       from the grid edge, where the sign of the derivative is chosen to make sure the
       extrapolation goes away from or towards the zero level set.

     For implicit surfaces, the extrapolation will typically be away from zero
       (the extrapolation should not imply the presence of an implicit surface
        beyond the array bounds).


    Input parameters
    ================
      dataIn (ndarray):	  Input data.
      dim (scalar):		    Dimension in which to add ghost nodes.
      width (scalar):	    Number of ghost nodes to add on each side (default = 1).
      ghostData (Bundle): Data structure containing data specific to this type of
                  ghost node.  For this function it is contains the field:

            .towardZero Boolean indicating whether sign of extrapolation should
                     be towards or away from the zero level set (default = 0).
    Output parameter
    ================
      dataOut (ndarray):	Output data.  
    """
    if isinstance(dataIn, np.ndarray):
      dataIn = cp.asarray(dataIn)

    if not width:
        width = 1

    if((width < 0) or (width > size(dataIn, dim))):
        error('Illegal width parameter')

    if(np.any(ghostData) and isinstance(ghostData, Bundle)):
        derivativeMultiplier = -1 if(ghostData.towardZero) else +1
    else:
        derivativeMultiplier = +1

    # create node array with array size
    dims = dataIn.ndim
    sizeIn = size(dataIn)
    indicesOut = []
    for i in range(dims):
        indicesOut.append(cp.arange(sizeIn[i], dtype=cp.intp))
    indicesIn = copy.copy(indicesOut)

    # create appropriately sized output array
    sizeOut = copy.copy(list(sizeIn))
    sizeOut[dim] = sizeOut[dim] + (2 * width)
    dataOut = cp.zeros(tuple(sizeOut), dtype=cp.float64)

    # fill output array with input data
    indicesOut[dim] = cp.arange(width, sizeOut[dim] - width, dtype=cp.intp) # correct
    # dynamic slicing to save the day
    dataOut[cp.ix_(*indicesOut)] = copy.copy(dataIn) 

    # compute derivatives
    indicesOut[dim] = [0]
    indicesIn[dim] = [1]
    derivativeBot = dataIn[cp.ix_(*indicesOut)] - dataIn[cp.ix_(*indicesIn)]

    indicesOut[dim] = [sizeIn[dim]-1]
    indicesIn[dim] = [sizeIn[dim] - 2]
    derivativeTop = dataIn[cp.ix_(*indicesOut)] - dataIn[cp.ix_(*indicesIn)]

    # adjust derivative sign to correspond with sign of data at array edge
    indicesIn[dim] = [0]
    derivativeBot = derivativeMultiplier * cp.abs(derivativeBot) * cp.sign(dataIn[cp.ix_(*indicesIn)])

    indicesIn[dim] = [sizeIn[dim]-1]
    derivativeTop = derivativeMultiplier * cp.abs(derivativeTop) * cp.sign(dataIn[cp.ix_(*indicesIn)])

    # now extrapolate
    for i in range(width):
        indicesOut[dim] = [i]
        indicesIn[dim] = [0]
        dataOut[cp.ix_(*indicesOut)] = (dataIn[cp.ix_(*indicesIn)] + (width - i) * derivativeBot)

        indicesOut[dim] = [sizeOut[dim]-1-i]
        indicesIn[dim] = [sizeIn[dim]-1]
        dataOut[cp.ix_(*indicesOut)] = (dataIn[cp.ix_(*indicesIn)] + (width - i) * derivativeTop)
        
    return dataOut
