import copy
import logging
logger = logging.getLogger(__name__)

from Utilities import *

def addGhostPeriodic(dataIn, dim, width=None, ghostData=None):
    """
     addGhostPeriodic: add ghost cells with periodic boundary conditions.

       dataOut = addGhostPeriodic(dataIn, dim, width, ghostData)

     creates ghost cells to manage the boundary conditions for the array dataIn

     this m-file fills the ghost cells with periodic data
       data from the top of the array is put in the bottom ghost cells
       data from the bottom of the array is put in the top ghost cells
       in 2D for dim == 1
              dataOut(1,1)   == dataIn(end+1-width,1)
              dataOut(end,1) == dataIn(width, 1)

     notice that the indexing is shifted by the ghost cell width in output array
       so in 2D for dim == 1, the first data in the original array will be at
              dataOut(width+1,1) == dataIn(1,1)

     parameters:
       dataIn	input data array
       dim		dimension in which to add ghost cells
       width	number of ghost cells to add on each side (default = 1)
       ghostData	A structure (see below).

       dataOut	Output data array.

     ghostData is a structure containing data specific to this type of
       ghost cell.  For this function it is entirely ignored.


     Copyright 2004 Ian M. Mitchell (mitchell@cs.ubc.ca).
     This software is used, copied and distributed under the licensing
       agreement contained in the file LICENSE in the top directory of
       the distribution.

     created Ian Mitchell, 5/12/03
     modified to allow choice of dimension, Ian Mitchell, 5/27/03
     modified to allow ghostData input structure, Ian Mitchell, 1/13/04

     Lekan Molu, Circa, August Week I, 2021
    """

    if not width:
      width = 1

    if((width < 0) or (width > dataIn.shape[dim])):
      logger.fatal('Illegal width parameter')

    # create cell array with array indices
    dims = ndims(dataIn)
    sizeIn = dataIn.shape
    indicesOut = cell(dims, 1)
    for i in range(dims):
      indicesOut[i] = list(range(sizeIn[i]))
    indicesIn = copy.copy(indicesOut)

    # create appropriately sized output array
    sizeOut = copy.copy(sizeIn)
    sizeOut[dim] = sizeOut[dim] + 2 * width
    dataOut = zeros(sizeOut, dtype=np.float64)

    # fill output array with input data
    indicesOut[dim] = width + list(range(sizeOut[dim])) - width
    dataOut[tuple(indicesOut[dim]),...] = dataIn

    # fill ghost cells
    indicesIn[dim] = (sizeIn[dim] - np.arange(width + 1,sizeIn[dim])).tolist()
    indicesOut[dim] = list(range(width))
    dataOut[indicesOut[dim]] = dataIn[indicesIn[dim]]

    indicesIn[dim] = list(range(width))
    indicesOut[dim] = (sizeIn[dim] - np.arange(width + 1,sizeIn[dim])).tolist()
    dataOut[indicesOut[dim]] = dataIn[indicesIn[dim]]


    return dataOut
