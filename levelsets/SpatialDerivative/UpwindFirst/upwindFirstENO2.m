function [ derivL, derivR ] = upwindFirstENO2(grid, data, dim, generateAll)
% upwindFirstENO2: second order upwind approx of first derivative.
%
%   [ derivL, derivR ] = upwindFirstENO2(grid, data, dim, generateAll)
%
% Computes a second order directional approximation to the first
%   derivative, using a oscillation reducing minimum modulus choice
%   of second order term.  The result is an order 2 ENO scheme.
%
% The approximation is constructed by a divided difference table.
%
% Some details of this scheme can be found in O&F, section 3.3,
%   where this scheme is equivalent to including the Q_1 and Q_2
%   terms of the ENO approximation.
%
% The generateAll option is used for debugging, and possibly by
%   higher order weighting schemes.  Under normal circumstances
%   the default (generateAll = false) should be used.
%
% parameters:
%   grid	Grid structure (see processGrid.m for details).
%   data        Data array.
%   dim         Which dimension to compute derivative on.
%   generateAll Return all possible second order upwind approximations.
%                 If this boolean is true, then derivL and derivR will
%                 be cell vectors containing all the approximations
%                 instead of just the minimum modulus approximation.
%                 (optional, default = 0)
%
%   derivL      Left approximation of first derivative (same size as data).
%   derivR      Right approximation of first derivative (same size as data).

% Copyright 2004 Ian M. Mitchell (mitchell@cs.ubc.ca).
% This software is used, copied and distributed under the licensing 
%   agreement contained in the file LICENSE in the top directory of 
%   the distribution.
%
% Ian Mitchell, 1/22/04

%---------------------------------------------------------------------------
if((dim < 0) | (dim > grid.dim))
  error('Illegal dim parameter');
end

if(nargin < 4)
  generateAll = 0;
end

dxInv = 1 / grid.dx(dim);

% How big is the stencil?
stencil = 2;

% Check that approximations that should be equivalent are equivalent 
%   (for debugging purposes, only used if generateAll == 1).
checkEquivalentApproximations = 1;
small = 100 * eps;             % a small number for "equivalence"

% Add ghost cells.
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
% First divided differences (first entry corresponds to D^1_{-1/2}).
indices1{dim} = 2 : size(gdata, dim);
indices2{dim} = indices1{dim} - 1;
D1 = dxInv * (gdata(indices1{:}) - gdata(indices2{:}));

% Second divided differences (first entry corresponds to D^2_0).
indices1{dim} = 2 : size(D1, dim);
indices2{dim} = indices1{dim} - 1;
D2 = 0.5 * dxInv * (D1(indices1{:}) - D1(indices2{:}));

%---------------------------------------------------------------------------
% First divided difference array has an extra entry at top and bottom
%   (from stencil width 2), so strip them off.
% Now first entry corresponds to D^1_{1/2}.
indices1{dim} = 2 : size(D1, dim) - 1;
D1 = D1(indices1{:});

%---------------------------------------------------------------------------
% First order approx is just the first order divided differences.
%   Make two copies to build the two approximations
dL = cell(2,1);
dR = cell(2,1);

% Take leftmost grid.N(dim) entries for left approximation.
indices1{dim} = 1 : size(D1, dim) - 1;
[ dL{:} ] = deal(D1(indices1{:}));

% Take rightmost grid.N(dim) entries for right approximation.
indices1{dim} = 2 : size(D1, dim);
[ dR{:} ] = deal(D1(indices1{:}));

%---------------------------------------------------------------------------
% Each copy gets modified by one of the second order terms.
%   Second order terms are sorted left to right.
indices1{dim} = 1 : size(D2, dim) - 2;
indices2{dim} = 2 : size(D2, dim) - 1;
dL{1} = dL{1} + grid.dx(dim) * D2(indices1{:});
dL{2} = dL{2} + grid.dx(dim) * D2(indices2{:});

indices1{dim} = indices1{dim} + 1;
indices2{dim} = indices2{dim} + 1;
dR{1} = dR{1} - grid.dx(dim) * D2(indices1{:});
dR{2} = dR{2} - grid.dx(dim) * D2(indices2{:});

%---------------------------------------------------------------------------
if(generateAll)

  if(checkEquivalentApproximations)
    % Rightward left and leftward right approximations should be the same
    %   (should be centered approximations, but we don't check for that).
    checkEquivalentApprox(dL{2}, dR{1}, small);    
  end

  % Caller requested both approximations in each direction.
  derivL = dL;
  derivR = dR;

%---------------------------------------------------------------------------
else

  % Need to figure out which approximation has the least oscillation.
  %   Note that L and R in this section refer to neighboring divided
  %   difference entries, not to left and right approximations.

  % Pick out minimum modulus neighboring D2 term.
  D2abs = abs(D2);
  indices1{dim} = 1 : size(D2, dim) - 1;
  indices2{dim} = indices1{dim} + 1;
  smallerL = (D2abs(indices1{:}) < D2abs(indices2{:}));
  smallerR = ~smallerL;
  
  %---------------------------------------------------------------------------
  % Pick out second order approximation that used the minimum modulus D2 term.
  indices1{dim} = 1 : size(smallerL, dim) - 1;
  derivL = dL{1} .* smallerL(indices1{:}) + dL{2} .* smallerR(indices1{:});
  
  indices1{dim} = 2 : size(smallerL, dim);
  derivR = dR{1} .* smallerL(indices1{:}) + dR{2} .* smallerR(indices1{:});

end
