__all__ = ["addGhostDirichlet"]

__author__ 		= "Lekan Molu"
__copyright__ 	= "2021, Hamilton-Jacobi Analysis in Python"
__credits__  	= "There are None."
__license__ 	= "MIT License"
__maintainer__ 	= "Lekan Molu"
__email__ 		= "patlekno@icloud.com"
__status__ 		= "Completed, Circa, August Week I, 2021."
__revised__     = "May 09, 2023"


import copy 
import logging
import cupy as cp
import numpy as np
from levelsetpy.utilities import *
logger = logging.getLogger(__name__)


def addGhostDirichlet(dataIn, dim, width=None, ghostData=None):
    """
        addGhostDirichlet: add ghost nodes with Dirichlet boundary conditions. The 
        Dirichlet boundary condition specifies the value of the solution to a partial
        differential equation or ordinary differential equation at the boundary of 
        the domain.

        Calling signature:

            dataOut = addGhostDirichlet(dataIn, dim, width, ghostData)

        The routine creates ghost nodes that manage the boundary conditions for the 
        array dataIn. This routine fills the ghost nodes with constant data, the solution
        to the differential equation at the boundary of the domain  (ie Dirichlet 
        boundary conditions).

        This routine only handles state (and time) independent
        Dirichlet data, though the value can be different on the upper
        and lower boundaries of the grid.

        Input parameters
        ================
            dataIn (ndarray):	Input data.
            dim (scalar):		Dimension in which to add ghost nodes.
            width (scalar):	    Number of ghost nodes to add on each side (default = 1).
            ghostData (Bundle): Data structure containing data specific to this type of
                    ghost node.  For this function it is contains the fields
                .lowerValue Scalar value to use on the lower side of the grid
                        (default = 0).
                .upperValue Scalar value to use on the upper side of the grid
                        (default = ghostData.lowerValue).
        Output parameter
        ================
        dataOut (ndarray):	Output data array.        
    """
    if not width:
        width = 1

    if((width < 0) or (width > size(dataIn, dim))):
        error('Illegal width parameter')

    if isbundle(ghostData):
        if isfield(ghostData, 'lowerValue'):
            lowerValue = ghostData.lowerValue
        else:
            error('ghostData bundle must contain the field, lowerValue.')

        if(isfield(ghostData, 'upperValue')):
            upperValue = ghostData.upperValue
        else:
            upperValue = lowerValue
    else:
        lowerValue = 0
        upperValue = copy.copy(lowerValue)

    dims = dataIn.ndim 
    sizeIn = dataIn.shape 
    indicesOut = [dim for dim in range(dims)]
    for i in range(dims):
        indicesOut[i] = range(sizeIn[i])
    indicesIn = copy.copy(indicesOut)

    sizeOut = copy.copy(sizeIn)
    sizeOut[dim] += 2*width 
    dataOut = cp.zeros(sizeOut)

    indicesOut[dim] = range(width, sizeOut[dim]-width)
    dataOut[cp.ix_(indicesOut)] = copy.copy(dataIn)
    
    indicesOut[dim] = range(width)
    dataOut[cp.ix_(indicesOut)] = lowerValue 
    
    indicesOut[dim] = range(sizeOut[dim] - width, sizeOut[dim])
    dataOut[cp.ix_(indicesOut)] = copy.copy(upperValue)
    
    return dataOut
