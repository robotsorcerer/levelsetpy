function dataOut = addGhostDirichlet(dataIn, dim, width, ghostData)
% addGhostDirichlet: add ghost cells with Dirichlet boundary conditions.
%
%   dataOut = addGhostDirichlet(dataIn, dim, width, ghostData)
%
% Creates ghost cells to manage the boundary conditions for the array dataIn.
%
% This m-file fills the ghost cells with constant data
%   (ie Dirichlet boundary conditions).
%
% At present, this code can only handle state (and time) independent
%   Dirichlet data, although the value can be different on the upper
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
%   .lowerValue Scalar value to use on the lower side of the grid
%		  (default = 0).
%   .upperValue Scalar value to use on the upper side of the grid
%		  (default = ghostData.lowerValue)
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
  if(isfield(ghostData, 'lowerValue'))
    lowerValue = ghostData.lowerValue;
  else
    error('ghostData structure must contain field lowerValue');
  end
  if(isfield(ghostData, 'upperValue'))
    upperValue = ghostData.upperValue;
  else
    upperValue = lowerValue;
  end
else
  lowerValue = 0;
  upperValue = lowerValue;
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

% now fill in the constants
indicesOut{dim} = 1 : width;
dataOut(indicesOut{:}) = lowerValue;

indicesOut{dim} = sizeOut(dim) - width + 1 : sizeOut(dim);
dataOut(indicesOut{:}) = upperValue;
