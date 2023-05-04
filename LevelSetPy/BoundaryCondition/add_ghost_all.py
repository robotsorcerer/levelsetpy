__all__ = ["addGhostAllDims"]

__author__ 		= "Lekan Molu"
__copyright__ 	= "2021, Hamilton-Jacobi Analysis in Python"
__credits__  	= "There are None."
__license__ 	= "Molux Licence"
__maintainer__ 	= "Lekan Molu"
__email__ 		= "patlekno@icloud.com"
__status__ 		= "Completed"

def   addGhostAllDims(grid, dataIn, width):
    """
     addGhostAllDims: Create ghost cells along all grid boundaries.

       dataOut = addGhostAllDims(grid, dataIn, width)

     Creates ghost cells to manage the boundary conditions for the array dataIn.

     This function adds the same number of ghost cells in every dimension
       according to the boundary conditions specified in the grid.

     Notice that the indexing is shifted by the ghost cell width in output array.
       So in 2D, the first data in the original array will be at
              dataOut(width+1,width+1) == dataIn(1,1)

     Parameters:
       grid	Grid structure (see processGrid.m for details).
       dataIn	Inp.t data array.
       width	Number of ghost cells to add on each side (default = 1).

       dataOut	Output data array.

     Lekan Aug 21, 2021
    """
    dataOut = dataIn

    # add ghost cells
    for i in range(grid.dim):
      dataOut = grid.bdry[i](dataOut, i, width, grid.bdryData[i])


    return dataOut
