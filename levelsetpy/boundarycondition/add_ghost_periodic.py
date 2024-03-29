__all__ = ["addGhostPeriodic"]

__author__ 		  = "Lekan Molu"
__copyright__ 	= "2021, Hamilton-Jacobi Analysis in Python"
__credits__  	  = "There are None."
__license__ 	  = "MIT License"
__maintainer__ 	= "Lekan Molu"
__email__ 		  = "patlekno@icloud.com"
__status__ 		  = "Completed, Circa, August Week I, 2021."
__revised__     = "May 09, 2023"

import copy
import cupy as cp
import numpy as np
import logging
logger = logging.getLogger(__name__)

from levelsetpy.utilities import *

def addGhostPeriodic(dataIn, dim, width=None, ghostData=None):
    """
     addGhostPeriodic: add ghost cells with periodic boundary conditions.

       dataOut = addGhostPeriodic(dataIn, dim, width, ghostData)

     creates ghost cells to manage the boundary conditions for the array dataIn

     this script fills the ghost cells with periodic data
       data from the top of the array is put in the bottom ghost cells
       data from the bottom of the array is put in the top ghost cells
       in 2D for dim == 1
              dataOut(1,1)   == dataIn(end+1-width,1)
              dataOut(end,1) == dataIn(width, 1)

     notice that the indexing is shifted by the ghost cell width in output array
       so in 2D for dim == 1, the first data in the original array will be at
              dataOut(width+1,1) == dataIn(1,1)

     Input parameters
     ================
       dataIn (ndarray):	Input data
       dim (scalar):		  Dimension in which to add ghost cells
       width (scalar):	  Number of ghost cells to add on each side (default = 1)
       ghostData (Bundle): Data structure containing data specific to this type of
            ghost node.  For this function it is entirely ignored.
    
     Output parameter
     ================
       dataOut:	Output data array.
    """
    if isinstance(dataIn, np.ndarray):
      dataIn = cp.asarray(dataIn)

    if not width:
      width = 1

    if((width < 0) or (width > dataIn.shape[dim])):
      logger.fatal('Illegal width parameter')

    # create cell array with array indices
    dims = dataIn.ndim
    sizeIn = dataIn.shape
    indicesOut = cell(dims)
    for i in range(dims):
      indicesOut[i] = index_array(1, sizeIn[i])
    indicesIn = copy.copy(indicesOut)

    # create appropriately sized output array
    sizeOut = copy.copy(list(sizeIn))
    sizeOut[dim] = sizeOut[dim] + 2 * width
    dataOut = cp.zeros(tuple(sizeOut), dtype=cp.float64)

    # fill output array with cp.intp data
    indicesOut[dim] = cp.arange(width, sizeOut[dim] - width, dtype=cp.intp)
    dataOut[cp.ix_(*indicesOut)] = dataIn

    # fill ghost cells
    indicesIn[dim] = cp.arange(sizeIn[dim] - width,sizeIn[dim], dtype=cp.intp)
    indicesOut[dim] = cp.arange(width, dtype=cp.intp)
    dataOut[cp.ix_(*indicesOut)] = dataIn[cp.ix_(*indicesIn)]

    indicesIn[dim] = cp.arange(width, dtype=cp.intp)
    indicesOut[dim] = cp.arange(sizeOut[dim] - width, sizeOut[dim], dtype=cp.intp)
    dataOut[cp.ix_(*indicesOut)] = dataIn[cp.ix_(*indicesIn)]

    return dataOut
