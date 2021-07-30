function [ vs, xs ] = addNodesAllDims(grid, width)
% addNodesAllDims: Create grid nodes corresponding to ghost cells.
%
%   [ vs, xs ] = addNodesAllDims(grid, width)
%
% Creates ghost nodes (ie an expansion of the grid.vs & grid.xs cell arrays) 
%   corresponding to the ghost cells created by addGhostAllDims.m.
%
% This function adds the same number of ghost nodes in every dimension.
%
% It is useful, for example, when you need to have the grid.xs arrays
%   for a computation including the ghost cells.
%
% Notice that the indexing is shifted by the ghost cell width in output array.
%   So in 2D, the first node in the original array will be at
%                  vs{i}(width+1) == grid.vs{i}(1)
%          xs{i}(width+1,width+1) == grid.xs{i}(1,1)
%
% Parameters:
%   grid	Grid structure (see processGrid.m for details).
%   width	Number of ghost cells to add on each side (default = 1).
%
%   xs		Cell vector, each element is an array
%		(the expanded version of grid.xs).
%   vs		Cell vector, each element is a vector
%		(the expanded version of grid.vs).
%

% Copyright 2004 Ian M. Mitchell (mitchell@cs.ubc.ca).
% This software is used, copied and distributed under the licensing 
%   agreement contained in the file LICENSE in the top directory of 
%   the distribution.
%
% created Ian Mitchell, 8/15/03

if(nargin < 2)
  width = 1;
end

grid = processGrid(grid);

% create the ghost node location vectors
vs = cell(grid.dim, 1);
for i = 1 : grid.dim
  vs{i} = [ grid.min(i) - (width : -1 : +1)' * grid.dx(i); ...
            grid.vs{i}; ...
            grid.max(i) + (+1 : +1 : width)' * grid.dx(i) ];
end

% ghost node arrays
if(nargout > 1)
  xs = cell(grid.dim, 1);
  [ xs{:} ] = ndgrid(vs{:});
end
