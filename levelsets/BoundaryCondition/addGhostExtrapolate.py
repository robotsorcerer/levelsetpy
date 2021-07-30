function dataOut = addGhostExtrapolate(dataIn, dim, width, ghostData)
% addGhostExtrapolate: add ghost cells, values extrapolated from bdry nodes.
%
%   dataOut = addGhostExtrapolate(dataIn, dim, width, ghostData)
%
% Creates ghost cells to manage the boundary conditions for the array dataIn.
%
% This m-file fills the ghost cells with data linearly extrapolated
%   from the grid edge, where the sign of the slope is chosen to make sure the
%   extrapolation goes away from or towards the zero level set.
%
% For implicit surfaces, the extrapolation will typically be away from zero
%   (the extrapolation should not imply the presence of an implicit surface
%    beyond the array bounds).
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
%   .towardZero Boolean indicating whether sign of extrapolation should
%                 be towards or away from the zero level set (default = 0).
%

% Copyright 2004 Ian M. Mitchell (mitchell@cs.ubc.ca).
% This software is used, copied and distributed under the licensing 
%   agreement contained in the file LICENSE in the top directory of 
%   the distribution.
%
% Ian Mitchell, 5/12/03
% modified to allow choice of dimension, Ian Mitchell, 5/27/03
% modified to allow ghostData input structure & renamed, Ian Mitchell, 1/13/04

if(nargin < 3)
  width = 1;
end

if((width < 0) || (width > size(dataIn, dim)))
  error('Illegal width parameter');
end

if((nargin == 4) && isstruct(ghostData))
  if(ghostData.towardZero)
    slopeMultiplier = -1;
  else
    slopeMultiplier = +1;
  end
else
  slopeMultiplier = +1;
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

% compute slopes
indicesOut{dim} = 1;
indicesIn{dim} = 2;
slopeBot = dataIn(indicesOut{:}) - dataIn(indicesIn{:});

indicesOut{dim} = sizeIn(dim);
indicesIn{dim} = sizeIn(dim) - 1;
slopeTop = dataIn(indicesOut{:}) - dataIn(indicesIn{:});

% adjust slope sign to correspond with sign of data at array edge
indicesIn{dim} = 1;
slopeBot = slopeMultiplier * abs(slopeBot) .* sign(dataIn(indicesIn{:}));
indicesIn{dim} = sizeIn(dim);
slopeTop = slopeMultiplier * abs(slopeTop) .* sign(dataIn(indicesIn{:}));

% now extrapolate
for i = 1 : width
  indicesOut{dim} = i;
  indicesIn{dim} = 1;
  dataOut(indicesOut{:}) = (dataIn(indicesIn{:}) + (width - i + 1) * slopeBot);

  indicesOut{dim} = sizeOut(dim) - i + 1;
  indicesIn{dim} = sizeIn(dim);
  dataOut(indicesOut{:}) = (dataIn(indicesIn{:}) + (width - i + 1) * slopeTop);
end
