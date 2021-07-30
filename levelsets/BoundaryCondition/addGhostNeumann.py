function dataOut = addGhostNeumann(dataIn, dim, width, ghostData)
% addGhostNeumann: add ghost cells with Neumann boundary conditions.
%
%   dataOut = addGhostNeumann(dataIn, dim, width, ghostData)
%
% Creates ghost cells to manage the boundary conditions for the array dataIn.
%
% This m-file fills the ghost cells with data of constant slope
%   (ie Neumann boundary conditions).
%
% At present, this code can only handle state (and time) independent
%   Neumann data, although the slope can be different on the upper
%   and lower boundaries of the grid.
%
% Notice that the indexing is shifted by the ghost cell width in output array.
%   So in 2D with dim == 1, the first data in the original array will be at
%          dataOut(width+1,1) == dataIn(1,1)
%
% parameters:
%   dataIn	Input data array.
%   dim		Dimension in which to add ghost cells.
%   width	Number of ghost cells to add on each side (default = 1).
%   ghostData	A structure (see below).
%
%   dataOut	Output data array.
%
% ghostData is a structure containing data specific to this type of
%   ghost cell.  For this function it contains the field(s)
%
%   .lowerSlope Scalar slope to use on the lower side of the grid
%		  assuming unit cell spacing, ie dx = 1 (default = 0).
%   .upperSlope Scalar slope to use on the upper side of the grid
%		  assuming unit cell spacing, ie dx = 1
%		  (default = ghostData.lowerSlope)
%

% Copyright 2004 Ian M. Mitchell (mitchell@cs.ubc.ca).
% This software is used, copied and distributed under the licensing 
%   agreement contained in the file LICENSE in the top directory of 
%   the distribution.
%
% Ian Mitchell, 1/13/04

if(nargin < 3)
  width = 1;
end

if((width < 0) || (width > size(dataIn, dim)))
  error('Illegal width parameter');
end

if((nargin == 4) && isstruct(ghostData))
  if(isfield(ghostData, 'lowerSlope'))
    lowerSlope = ghostData.lowerSlope;
  else
    error('ghostData structure must contain field lowerSlope');
  end
  if(isfield(ghostData, 'upperSlope'))
    upperSlope = ghostData.upperSlope;
  else
    upperSlope = lowerSlope;
  end
else
  lowerSlope = 0;
  upperSlope = lowerSlope;
end

% create cell array with array size
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

% now extrapolate
for i = 1 : width
  indicesOut{dim} = i;
  indicesIn{dim} = 1;
  dataOut(indicesOut{:}) = (dataIn(indicesIn{:}) + (width - i + 1)*lowerSlope);

  indicesOut{dim} = sizeOut(dim) - i + 1;
  indicesIn{dim} = sizeIn(dim);
  dataOut(indicesOut{:}) = (dataIn(indicesIn{:}) + (width - i + 1)*upperSlope);
end
