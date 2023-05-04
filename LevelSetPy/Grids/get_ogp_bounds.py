__all__ = ["getOGPBounds"]

__author__ 		= "Lekan Molu"
__copyright__ 	= "2021, Hamilton-Jacobi Analysis in Python"
__credits__  	= "There are None."
__license__ 	= "MIT License"
__maintainer__ 	= "Lekan Molu"
__email__ 		= "patlekno@icloud.com"
__status__ 		= "Completed"


import numpy as np
from LevelSetPy.Utilities import *

def  getOGPBounds(gBase, gMinIn, gMaxIn, padding):
    """"
        [gMinOut, gMaxOut] = getOGPBounds(gBase, gMinIn, gMaxIn)
        Returns grid bounds based on gBase, gMinIn, and gMaxIn such that if a new
        grid is constructed from gMinOut and gMaxOut, the grid points within the
        bounds of gBase would be the same.

        padding must be same dim as gMinIn, gMaxIn

        This is done without needing the actual grid points of gBase
    """

    # Compute or read grid spacing
    if isfield(gBase, 'dx'):
        dx = gBase.dx
    else:
        dx = np.divide((gBase.max - gBase.min), (gBase.N))

    # Add padding to both sides
    #print(f'gMinIn: {gMinIn.shape} padding {padding.shape}')
    gMinIn -= padding
    gMaxIn += padding

    # Initialize
    gMaxOut = zeros(gBase.dim, 1, dtype=np.float64)
    gMinOut = zeros(gBase.dim, 1, dtype=np.float64)
    NOut = zeros(gBase.dim, 1)

    for dim  in range(gBase.dim):
        # Arbitrary reference point
        refGridPt = gBase.min[dim]

        # Get minimum and maximum bounds for this dimension
        ptrMax = np.floor((gMaxIn[dim] - refGridPt) / dx[dim])
        gMaxOut[dim] = refGridPt + ptrMax*dx[dim]

        ptrMin = np.ceil((gMinIn[dim] - refGridPt) / dx[dim])
        gMinOut[dim] = refGridPt + ptrMin*dx[dim]

        # Get number of grid points
        NOut[dim] = int(ptrMax - ptrMin + 1)

    return gMinOut, gMaxOut, NOut
