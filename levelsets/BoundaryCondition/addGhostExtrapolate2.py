function dataOut = addGhostExtrapolate2(dataIn, dim, width, ghostData)
% addGhostExtrapolate2: add ghost cells, order 2 extrapolation from bdry nodes.
%
%   dataOut = addGhostExtrapolate2(dataIn, dim, width, ghostData)
%
% Creates ghost cells to manage the boundary conditions for the array dataIn.
%
% This m-file fills the ghost cells with data extrapolated by a quadratic
%   approximation from the grid edge.
%
% At present, the extrapolation is done with no regard to whether the
%   implicit surface function is bending toward or away from a (nonexistant)
%   implicit surface beyond the computational domain boundary.
%
% It would be nice to find a theoretically justified fix,
%   which would hopefully also explain why the sign corrected extrapolation 
%   in addGhostExtrapolate works so well.
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
%   ghost cell.  For this function it is entirely ignored.

% Copyright 2004 Ian M. Mitchell (mitchell@cs.ubc.ca).
% This software is used, copied and distributed under the licensing 
%   agreement contained in the file LICENSE in the top directory of 
%   the distribution.
%
% Ian Mitchell, 8/27/04

%---------------------------------------------------------------------------
% Check input parameters.

if(nargin < 3)
  width = 1;
end

if((width < 0) || (width > size(dataIn, dim)))
  error('Illegal width parameter');
end

%---------------------------------------------------------------------------
% This code might prove useful if we can find some way to extrapolate
%   away from (or towards) the zero level set.
if(0)
   if((nargin == 4) && isstruct(ghostData))
     if(ghostData.towardZero)
      slopeMultiplier = -1;
    else
      slopeMultiplier = +1;
    end
  else
    slopeMultiplier = +1;
  end
end

%---------------------------------------------------------------------------
% Create cell arrays with array sizes for indexing.
dims = ndims(dataIn);
sizeIn = size(dataIn);
indicesOut = cell(dims, 1);
for i = 1 : dims
  indicesOut{i} = 1:sizeIn(i);
end
indicesMid = indicesOut;
indicesIn = indicesOut;

%---------------------------------------------------------------------------
% Create appropriately sized output array.
sizeOut = sizeIn;
sizeOut(dim) = sizeOut(dim) + 2 * width;
dataOut = zeros(sizeOut);

% Fill output array with input data from the interior of the domain.
indicesOut{dim} = width + 1 : sizeOut(dim) - width;
dataOut(indicesOut{:}) = dataIn;

%---------------------------------------------------------------------------
% Compute slopes (first order approximations).
indicesOut{dim} = 1;
indicesIn{dim} = 2;
slopeBot = dataIn(indicesOut{:}) - dataIn(indicesIn{:});

indicesOut{dim} = sizeIn(dim);
indicesIn{dim} = sizeIn(dim) - 1;
slopeTop = dataIn(indicesOut{:}) - dataIn(indicesIn{:});

%---------------------------------------------------------------------------
% Compute second order approximation.
indicesOut{dim} = 1;
indicesMid{dim} = 2;
indicesIn{dim} = 3;
secondBot = dataIn(indicesOut{:}) - 2 * dataIn(indicesMid{:}) ...
            + dataIn(indicesIn{:});

indicesOut{dim} = sizeIn(dim);
indicesMid{dim} = sizeIn(dim) - 1;
indicesIn{dim} = sizeIn(dim) - 2;
secondTop = dataIn(indicesOut{:}) - 2 * dataIn(indicesMid{:}) ...
            + dataIn(indicesIn{:});

%---------------------------------------------------------------------------
% Adjust slope sign to correspond with sign of data at array edge.
% This code might prove useful if we can find some way to extrapolate
%   away from (or towards) the zero level set.
if(0)
  indicesIn{dim} = 1;
  slopeBot = slopeMultiplier * abs(slopeBot) .* sign(dataIn(indicesIn{:}));
  indicesIn{dim} = sizeIn(dim);
  slopeTop = slopeMultiplier * abs(slopeTop) .* sign(dataIn(indicesIn{:}));
end

%---------------------------------------------------------------------------
% Extrapolate to fill in ghost cells.
for i = 1 : width
  gap = width - i + 1;

  indicesOut{dim} = i;
  indicesIn{dim} = 1;
  dataOut(indicesOut{:}) = (dataIn(indicesIn{:}) + gap * slopeBot ...
                            + 0.5 * gap * (gap + 1) * secondBot);

  indicesOut{dim} = sizeOut(dim) - i + 1;
  indicesIn{dim} = sizeIn(dim);
  dataOut(indicesOut{:}) = (dataIn(indicesIn{:}) + gap * slopeTop ...
                            + 0.5 * gap * (gap + 1) * secondTop);
end
