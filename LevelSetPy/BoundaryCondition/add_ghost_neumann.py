__all__ = ["addGhostNeumann"]

__author__ 		    = "Lekan Molu"
__copyright__ 	    = "2021, Hamilton-Jacobi Analysis in Python"
__credits__  	    = "There are None."
__license__ 	    = "MIT License"
__maintainer__ 	    = "Lekan Molu"
__email__ 		    = "patlekno@icloud.com"
__status__ 		    = "Completed, Circa, August Week I, 2021."
__revised__         = "May 09, 2023"


import copy 
import logging
import cupy as cp
import numpy as np
from LevelSetPy.Utilities import *
logger = logging.getLogger(__name__)


def addGhostNeumann(dataIn, dim, width=None, ghostData=None):
    """
        addGhostNeumann: add ghost cells with Neumann boundary conditions. The 
        Neumann boundary condition specifies the derivative of the solution to a partial
        differential equation or ordinary differential equation at the boundary of 
        the domain.

        Calling signature:

            dataOut = addGhostNeumann(dataIn, dim, width, ghostData)

        The routine creates ghost cells that manage the boundary conditions for the 
        array dataIn. 

        This routine only handles state (and time) independent
        Neumann data, though the value can be different on the upper
        and lower boundaries of the grid.

        Input parameters
        ================
        dataIn (ndarray):	Input data
        dim (scalar):		Dimension in which to add ghost cells
        width (scalar):	    Number of ghost cells to add on each side (default = 1)
        ghostData (Bundle): Data structure containing data specific to this type of
                ghost node.  For this function it is contains the fields
            .lowerDerivative Scalar derivative to use on the lower side of the grid
                    assuming unit node spacing, ie dx = 1 (default = 0).
            .upperDerivative Scalar derivative to use on the upper side of the grid
                    assuming unit node spacing, ie dx = 1 (default = ghostData.lowerDerivative).
        Output parameter
        ================
        dataOut (ndarray):	Output data array.
    """
    if not width:
        width = 1

    if((width < 0) or (width > size(dataIn, dim))):
        error('Illegal width parameter')

    if isbundle(ghostData):
        if isfield(ghostData, 'lowerDerivative'):
            lowerDerivative = ghostData.lowerDerivative
        else:
            error('ghostData bundle must contain the field, lowerDerivative.')

        if(isfield(ghostData, 'upperDerivative')):
            upperDerivative = ghostData.upperDerivative
        else:
            upperDerivative = lowerDerivative
    else:
        lowerDerivative = 0
        upperDerivative = copy.copy(lowerDerivative)

    dims = dataIn.ndim 
    sizeIn = dataIn.shape 
    indicesOut = [dim for dim in range(dims)]
    for dim in range(dims):
        indicesOut[dim] = range(sizeIn[dim])
    indicesIn = copy.copy(indicesOut)

    sizeOut       = copy.copy(sizeIn)
    sizeOut[dim] += 2*width 
    dataOut       = cp.zeros(sizeOut)

    indicesOut[dim] = range(width, sizeOut[dim]-width)
    dataOut[cp.ix_(*indicesOut)] = copy.copy(dataIn)
    
    # extrapolate 
    for i in range(width):
        indicesOut[dim] = i
        indicesIn[dim]  = 1
        dataOut[cp.ix_(*indicesOut)] = dataIn[cp.ix_(*indicesIn)] + (width - i) * lowerDerivative
    
        indicesOut[dim] = copy.copy(sizeOut[dim] - i - 1)
        indicesIn[dim]  = copy.copy(sizeIn[dim] )
        dataOut[cp.ix_(*indicesOut)] = dataIn[cp.ix_(*indicesIn)] + (width - i) * upperDerivative
    
    return dataOut