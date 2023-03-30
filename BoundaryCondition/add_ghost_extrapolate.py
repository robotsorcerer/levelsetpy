__author__ 		= "Lekan Molu"
__maintainer__ 	= "Lekan Molu"
__email__ 		= "patlekno@icloud.com"
__status__ 		= "Completed"

__all__ = ["addGhostExtrapolate"]


import copy
import logging
import cupy as cp
import numpy as np
from LevelSetPy.Utilities import *
logger = logging.getLogger(__name__)

def addGhostExtrapolate(dataIn, dim, width=None, ghostData=None):
    """
     addGhostExtrapolate: add ghost cells, values extrapolated from bdry nodes.

       dataOut = addGhostExtrapolate(dataIn, dim, width, ghostData)

     Creates ghost cells to manage the boundary conditions for the array dataIn.

     This script fills the ghost cells with data linearly extrapolated
       from the grid edge, where the sign of the slope is chosen to make sure the
       extrapolation goes away from or towards the zero level set.

     For implicit surfaces, the extrapolation will typically be away from zero
       (the extrapolation should not imply the presence of an implicit surface
        beyond the array bounds).

     Notice that the indexing is shifted by the ghost cell width in output array.
       So in 2D with dim == 1, the first data in the original array will be at
              dataOut(width+1,1) == dataIn(1,1)

     parameters:
       dataIn	cp.intp data array.
       dim		Dimension in which to add ghost cells.
       width	Number of ghost cells to add on each side (default = 1).
       ghostData	A structure (see below).

       dataOut	Output data array.

     ghostData is a structure containing data specific to this type of
       ghost cell.  For this function it contains the field(s)

       .towardZero Boolean indicating whether sign of extrapolation should
                     be towards or away from the zero level set (default = 0).

     Lekan Molu, Circa, August Week I, 2021
    """
    if isinstance(dataIn, np.ndarray):
      dataIn = cp.asarray(dataIn)

    if not width:
        width = 1

    if((width < 0) or (width > size(dataIn, dim))):
        error('Illegal width parameter')

    if(np.any(ghostData) and isinstance(ghostData, Bundle)):
        slopeMultiplier = -1 if(ghostData.towardZero) else +1
    else:
        slopeMultiplier = +1

    # create cell array with array size
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

    # fill output array with icp.t data
    indicesOut[dim] = cp.arange(width, sizeOut[dim] - width, dtype=cp.intp) # correct
    # dynamic slicing to save the day
    dataOut[cp.ix_(*indicesOut)] = copy.copy(dataIn) # correct
    #logger.debug(f'dataIn: {cp.linalg.norm(dataIn, 2)} dataOut: {cp.linalg.norm(dataOut, 2)}')

    # compute slopes
    indicesOut[dim] = [0]
    indicesIn[dim] = [1]
    slopeBot = dataIn[cp.ix_(*indicesOut)] - dataIn[cp.ix_(*indicesIn)]

    indicesOut[dim] = [sizeIn[dim]-1]
    indicesIn[dim] = [sizeIn[dim] - 2]
    slopeTop = dataIn[cp.ix_(*indicesOut)] - dataIn[cp.ix_(*indicesIn)]
    #logger.debug(f'slopeBot: {cp.linalg.norm(slopeBot, 2)} slopeTop: {cp.linalg.norm(slopeTop, 2)}')

    # adjust slope sign to correspond with sign of data at array edge
    indicesIn[dim] = [0]
    slopeBot = slopeMultiplier * cp.abs(slopeBot) * cp.sign(dataIn[cp.ix_(*indicesIn)])

    indicesIn[dim] = [sizeIn[dim]-1]
    slopeTop = slopeMultiplier * cp.abs(slopeTop) * cp.sign(dataIn[cp.ix_(*indicesIn)])

    # now extrapolate
    for i in range(width):
        indicesOut[dim] = [i]
        indicesIn[dim] = [0]
        dataOut[cp.ix_(*indicesOut)] = (dataIn[cp.ix_(*indicesIn)] + (width - i) * slopeBot)

        indicesOut[dim] = [sizeOut[dim]-1-i]
        indicesIn[dim] = [sizeIn[dim]-1]
        dataOut[cp.ix_(*indicesOut)] = (dataIn[cp.ix_(*indicesIn)] + (width - i) * slopeTop)

    cp.cuda.Device().synchronize()
    return dataOut
