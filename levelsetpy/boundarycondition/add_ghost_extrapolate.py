
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
import torch
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
      dataIn = torch.as_tensor(dataIn)

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
        indicesOut.append(torch.arange(sizeIn[i], dtype=torch.int64))
    indicesIn = copy.copy(indicesOut)

    # create appropriately sized output array
    sizeOut = copy.copy(list(sizeIn))
    sizeOut[dim] = sizeOut[dim] + (2 * width)
    dataOut = torch.zeros(tuple(sizeOut), dtype=DTYPE)

    # fill output array with input data
    indicesOut[dim] = torch.arange(width, sizeOut[dim] - width, dtype=torch.int64) # correct
    # dynamic slicing to save the day
    dataOut[torch.meshgrid(*indicesOut, indexing='ij')] = copy.copy(dataIn)

    # compute derivatives
    indicesOut[dim] = torch.tensor([0], dtype=torch.int64)
    indicesIn[dim] = torch.tensor([1], dtype=torch.int64)
    derivativeBot = dataIn[torch.meshgrid(*indicesOut, indexing='ij')] - dataIn[torch.meshgrid(*indicesIn, indexing='ij')]

    indicesOut[dim] = torch.tensor([sizeIn[dim]-1], dtype=torch.int64)
    indicesIn[dim] = torch.tensor([sizeIn[dim] - 2], dtype=torch.int64)
    derivativeTop = dataIn[torch.meshgrid(*indicesOut, indexing='ij')] - dataIn[torch.meshgrid(*indicesIn, indexing='ij')]

    # adjust derivative sign to correspond with sign of data at array edge
    indicesIn[dim] = torch.tensor([0], dtype=torch.int64)
    derivativeBot = derivativeMultiplier * torch.abs(derivativeBot) * torch.sign(dataIn[torch.meshgrid(*indicesIn, indexing='ij')])

    indicesIn[dim] = torch.tensor([sizeIn[dim]-1], dtype=torch.int64)
    derivativeTop = derivativeMultiplier * torch.abs(derivativeTop) * torch.sign(dataIn[torch.meshgrid(*indicesIn, indexing='ij')])

    # now extrapolate
    for i in range(width):
        indicesOut[dim] = torch.tensor([i], dtype=torch.int64)
        indicesIn[dim] = torch.tensor([0], dtype=torch.int64)
        dataOut[torch.meshgrid(*indicesOut, indexing='ij')] = (dataIn[torch.meshgrid(*indicesIn, indexing='ij')] + (width - i) * derivativeBot)

        indicesOut[dim] = torch.tensor([sizeOut[dim]-1-i], dtype=torch.int64)
        indicesIn[dim] = torch.tensor([sizeIn[dim]-1], dtype=torch.int64)
        dataOut[torch.meshgrid(*indicesOut, indexing='ij')] = (dataIn[torch.meshgrid(*indicesIn, indexing='ij')] + (width - i) * derivativeTop)
        
    return dataOut
