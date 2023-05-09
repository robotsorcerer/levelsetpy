__all__ = ["addGhostAllDims"]

__author__ 		  = "Lekan Molu"
__copyright__ 	= "2021, Hamilton-Jacobi Analysis in Python"
__credits__  	  = "There are None."
__license__ 	  = "MIT License"
__maintainer__ 	= "Lekan Molu"
__email__ 		  = "patlekno@icloud.com"
__status__ 		  = "Completed, Circa, August Week I, 2021."
__revised__     = "May 09, 2023."

import copy

def   addGhostAllDims(grid, dataIn, width):
    """
     addGhostAllDims: Create ghost nodes along all grid boundaries.

       dataOut = addGhostAllDims(grid, dataIn, width)

     Creates ghost nodes to manage the boundary conditions for the array dataIn.

     This function adds the same number of ghost nodes in every dimension
       according to the boundary conditions specified in the grid.

     Notice that the indexing is shifted by the ghost node width in output array.
       So in 2D, the first data in the original array will be at
              dataOut(width+1,width+1) == dataIn(1,1)


      Input parameters
      ================
          dataIn (ndarray):	Input data.
          dim (scalar):		Dimension in which to add ghost nodes.
          width (scalar):	    Number of ghost nodes to add on each side (default = 1).

      Output parameter
      ================
      dataOut (ndarray):	Output data array.  
    """
    dataOut = copy.copy(dataIn)

    for i in range(grid.dim):
      dataOut = grid.bdry[i](dataOut, i, width, grid.bdryData[i])


    return dataOut
