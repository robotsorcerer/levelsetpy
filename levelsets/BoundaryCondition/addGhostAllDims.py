function dataOut = addGhostAllDims(grid, dataIn, width)
% addGhostAllDims: Create ghost cells along all grid boundaries.
%
%   dataOut = addGhostAllDims(grid, dataIn, width)
%
% Creates ghost cells to manage the boundary conditions for the array dataIn.
%
% This function adds the same number of ghost cells in every dimension
%   according to the boundary conditions specified in the grid.
%
% Notice that the indexing is shifted by the ghost cell width in output array.
%   So in 2D, the first data in the original array will be at
%          dataOut(width+1,width+1) == dataIn(1,1)
%
% Parameters:
%   grid	Grid structure (see processGrid.m for details).
%   dataIn	Input data array.
%   width	Number of ghost cells to add on each side (default = 1).
%
%   dataOut	Output data array.
%

% Copyright 2004 Ian M. Mitchell (mitchell@cs.ubc.ca).
% This software is used, copied and distributed under the licensing 
%   agreement contained in the file LICENSE in the top directory of 
%   the distribution.
%
% Ian Mitchell, 6/3/03

dataOut = dataIn;

% add ghost cells
for i = 1 : grid.dim
  dataOut = feval(grid.bdry{i}, dataOut, i, width, grid.bdryData{i});
end
