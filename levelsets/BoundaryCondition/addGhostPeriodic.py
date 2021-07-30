function dataOut = addGhostPeriodic(dataIn, dim, width, ghostData)
% addGhostPeriodic: add ghost cells with periodic boundary conditions.
%
%   dataOut = addGhostPeriodic(dataIn, dim, width, ghostData)
%
% creates ghost cells to manage the boundary conditions for the array dataIn
%
% this m-file fills the ghost cells with periodic data
%   data from the top of the array is put in the bottom ghost cells
%   data from the bottom of the array is put in the top ghost cells
%   in 2D for dim == 1
%          dataOut(1,1)   == dataIn(end+1-width,1)
%          dataOut(end,1) == dataIn(width, 1)
%
% notice that the indexing is shifted by the ghost cell width in output array
%   so in 2D for dim == 1, the first data in the original array will be at
%          dataOut(width+1,1) == dataIn(1,1)
%
% parameters:
%   dataIn	input data array
%   dim		dimension in which to add ghost cells
%   width	number of ghost cells to add on each side (default = 1)
%   ghostData	A structure (see below).
%
%   dataOut	Output data array.
%
% ghostData is a structure containing data specific to this type of
%   ghost cell.  For this function it is entirely ignored.
%

% Copyright 2004 Ian M. Mitchell (mitchell@cs.ubc.ca).
% This software is used, copied and distributed under the licensing 
%   agreement contained in the file LICENSE in the top directory of 
%   the distribution.
%
% created Ian Mitchell, 5/12/03
% modified to allow choice of dimension, Ian Mitchell, 5/27/03
% modified to allow ghostData input structure, Ian Mitchell, 1/13/04

if(nargin < 3)
  width = 1;
end

if((width < 0) | (width > size(dataIn, dim)))
  error('Illegal width parameter');
end

% create cell array with array indices
dims = ndims(dataIn);
sizeIn = size(dataIn);
indicesOut = cell(dims, 1);
for i = 1 : dims
  indicesOut{i} = 1:sizeIn(i);
end
indicesIn = indicesOut;

% create appropriately sized output array
sizeOut = sizeIn;
sizeOut(dim) = sizeOut(dim) + 2 * width;
dataOut = zeros(sizeOut);

% fill output array with input data
indicesOut{dim} = width + 1 : sizeOut(dim) - width;
dataOut(indicesOut{:}) = dataIn;

% fill ghost cells
indicesIn{dim} = sizeIn(dim) - width + 1 : sizeIn(dim);
indicesOut{dim} = 1 : width;
dataOut(indicesOut{:}) = dataIn(indicesIn{:});

indicesIn{dim} = 1 : width;
indicesOut{dim} = sizeOut(dim) - width + 1 : sizeOut(dim);
dataOut(indicesOut{:}) = dataIn(indicesIn{:});
