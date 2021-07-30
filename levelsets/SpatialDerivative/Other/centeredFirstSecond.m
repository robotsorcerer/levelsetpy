function deriv = centeredFirstSecond(grid, data, dim)
% centeredFirstSecond: second order centered difference approx of first deriv.
%
%   deriv = centeredFirstSecond(grid, data, dim)
%
% Computes a second order centered difference approximation to the
%   first derivative in dimension dim.
%
% Note that centered differences are not a good choice for first order
%   terms in Hamilton-Jacobi equations (such as advection).  Instead
%   use an upwind scheme.
%
% parameters:
%   grid	Grid structure (see processGrid.m for details).
%   data        Data array.
%   dim         Which dimension to compute derivative on.
%
%   deriv       Centered approximation of first derivative (same size as data).

% Copyright 2004 Ian M. Mitchell (mitchell@cs.ubc.ca).
% This software is used, copied and distributed under the licensing 
%   agreement contained in the file LICENSE in the top directory of 
%   the distribution.
%
% Ian Mitchell, 02/02/03

%---------------------------------------------------------------------------
if((dim < 0) | (dim > grid.dim))
  error('Illegal dim parameter');
end

dxInv = 1 ./ grid.dx(dim);

% How big is the stencil?
stencil = 1;

% Add ghost cells to every dimension.
gdata = feval(grid.bdry{dim}, data, dim, stencil, grid.bdryData{dim});

%---------------------------------------------------------------------------
% Create cell array with array indices.
sizeData = size(gdata);
indices1 = cell(grid.dim, 1);
for i = 1 : grid.dim
  indices1{i} = 1:sizeData(i);
end
indices2 = indices1;

%---------------------------------------------------------------------------
% Take a difference in the appropriate dimension.
indices1{dim} = 1 : size(gdata, dim) - 2;
indices2{dim} = 3 : size(gdata, dim);
deriv = 0.5 * dxInv * (gdata(indices2{:}) - gdata(indices1{:}));
