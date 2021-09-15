import copy
from Utilities import *

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
       dataIn	Input data array.
       dim		Dimension in which to add ghost cells.
       width	Number of ghost cells to add on each side (default = 1).
       ghostData	A structure (see below).

       dataOut	Output data array.

     ghostData is a structure containing data specific to this type of
       ghost cell.  For this function it contains the field(s)

       .towardZero Boolean indicating whether sign of extrapolation should
                     be towards or away from the zero level set (default = 0).


     Copyright 2004 Ian M. Mitchell (mitchell@cs.ubc.ca).
     This software is used, copied and distributed under the licensing
       agreement contained in the file LICENSE in the top directory of
       the distribution.

     Ian Mitchell, 5/12/03
     modified to allow choice of dimension, Ian Mitchell, 5/27/03
     modified to allow ghostData input structure & renamed, Ian Mitchell, 1/13/04

     Lekan Molu, Circa, August Week I, 2021
    """
    if not width:
        width = 1

    if((width < 0) or (width > size(dataIn, dim))):
        error('Illegal width parameter')

    if(ghostData and isinstance(ghostData, Bundle)):
        slopeMultiplier = -1 if(ghostData.towardZero) else +1
    else:
        slopeMultiplier = +1

    # create cell array with array size
    dims = ndims(dataIn)
    sizeIn = size(dataIn)
    indicesOut = []
    for i in range(dims):
        indicesOut.append(np.arange(sizeIn[i], dtype=np.intp))
    indicesIn = copy.copy(indicesOut)

    # create appropriately sized output array
    sizeOut = copy.copy(list(sizeIn))

    sizeOut[dim] = sizeOut[dim] + (2 * width)
    dataOut = zeros(tuple(sizeOut), dtype=np.float64)

    # fill output array with input data
    indicesOut[dim] = np.arange(width, sizeOut[dim] - width, dtype=np.intp) # correct
    # dynamic slicing to save the day
    dataOut[np.ix_(*indicesOut)] = dataIn # correct

    # compute slopes
    indicesOut[dim] = [0]
    indicesIn[dim] = [1]
    slopeBot = dataIn[np.ix_(*indicesOut)] - dataIn[np.ix_(*indicesIn)]

    indicesOut[dim] = [sizeIn[dim]-1]
    indicesIn[dim] = [sizeIn[dim] - 2]
    slopeTop = dataIn[np.ix_(*indicesOut)] - dataIn[np.ix_(*indicesIn)]

    # adjust slope sign to correspond with sign of data at array edge
    indicesIn[dim] = [0]
    slopeBot = slopeMultiplier * np.abs(slopeBot) * np.sign(dataIn[np.ix_(*indicesIn)])
    indicesIn[dim] = [sizeIn[dim]-1] # account for python/C indexing
    slopeTop = slopeMultiplier * np.abs(slopeTop) * np.sign(dataIn[np.ix_(*indicesIn)])

    # now extrapolate
    for i in range(width):
        indicesOut[dim] = [i]
        indicesIn[dim] = [0]
        dataOut[np.ix_(*indicesOut)] = (dataIn[np.ix_(*indicesIn)] + (width - i + 1) * slopeBot)

        if i == 0:
            indicesOut[dim] = [sizeOut[dim] - i -1]
        else:
            indicesOut[dim] = [sizeOut[dim] - i]
        indicesIn[dim] = [sizeIn[dim]-1]
        dataOut[np.ix_(*indicesOut)] = (dataIn[np.ix_(*indicesIn)] + (width - i + 1) * slopeTop)

    return dataOut
