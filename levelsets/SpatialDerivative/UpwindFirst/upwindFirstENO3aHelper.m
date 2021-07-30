function [ dL, dR, DD ] = ...
    upwindFirstENO3aHelper(grid, data, dim, approx4, stripDD)
% upwindFirstENO3aHelper: helper function for upwindFirstENO3a.
%
%   [ dL, dR, DD ] = upwindFirstENO3aHelper(grid, data, dim, approx4, stripDD)
%
% Helper function to compute the ENO and WENO directional approximations
%   to the first derivative using a divided difference table.
%
% The approximation is constructed by a divided difference table,
%   which is more efficient (although a little more complicated)
%   than using the direct equations from O&F section 3.4
%   (see upwindFirstENO3b for that version).
%
% Details of this scheme can be found in O&F, section 3.3,
%   where this scheme is equivalent to including the Q_1, Q_2 and Q_3
%   terms of the ENO approximation.
%
% parameters:
%   grid	Grid structure (see processGrid.m for details).
%   data        Data array.
%   dim         Which dimension to compute derivative on.
%   approx4     Generate two copies of middle approximation using
%                 both left/right and right/left traversal of divided
%                 difference tree.  The extra copy is placed in the
%                 fourth element of derivL and derivR, and is equivalent
%                 to the version in the second element of those cell vectors.
%   stripDD     Strip the divided difference tables down to their
%                 appropriate size, otherwise they will contain entries
%                 (at the D1 and D2 levels) that correspond entirely
%                 to ghost cells.
%
%   dL          Cell vector containing the 3 or 4 left approximations 
%                 of the first derivative (each the same size as data).
%   dR          Cell vector containing the 3 or 4 right approximations 
%                 of the first derivative (each the same size as data).
%   DD          Cell vector containing the divided difference tables
%                 (optional).

% Copyright 2004 Ian M. Mitchell (mitchell@cs.ubc.ca).
% This software is used, copied and distributed under the licensing 
%   agreement contained in the file LICENSE in the top directory of 
%   the distribution.
%
% Ian Mitchell, 1/26/04

%---------------------------------------------------------------------------
dxInv = 1 / grid.dx(dim);

% How big is the stencil?
stencil = 3;

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
% First divided differences (first entry corresponds to D^1_{-3/2}).
indices1{dim} = 2 : size(gdata, dim);
indices2{dim} = indices1{dim} - 1;
D1 = dxInv * (gdata(indices1{:}) - gdata(indices2{:}));

% Second divided differences (first entry corresponds to D^2_{-1}).
indices1{dim} = 2 : size(D1, dim);
indices2{dim} = indices1{dim} - 1;
D2 = 0.5 * dxInv * (D1(indices1{:}) - D1(indices2{:}));

% Third divided differences (first entry corresponds to D^3_{-1/2}).
indices1{dim} = 2 : size(D2, dim);
indices2{dim} = indices1{dim} - 1;
D3 = (1/3) * dxInv * (D2(indices1{:}) - D2(indices2{:}));

%---------------------------------------------------------------------------
% If we want the unstripped divided difference entries, make a copy now.
if((nargout > 2) && ~stripDD)
  DD = { D1; D2; D3 };
end

% First divided difference array has 2 extra entries at top and bottom
%   (from stencil width 3), so strip them off.
% Now first entry corresponds to D^1_{1/2}.
indices1{dim} = 3 : size(D1, dim) - 2;
D1 = D1(indices1{:});

% Second divided difference array has an extra entry at top and bottom
%   (from stencil width 3), so strip them off.
% Now first entry corresponds to D^2_0.
indices1{dim} = 2 : size(D2, dim) - 1;
D2 = D2(indices1{:});

% If we want the stripped divided difference entries, make a copy now.
if((nargout > 2) && stripDD)
  DD = { D1; D2; D3 };
end

%---------------------------------------------------------------------------
% First order approx is just the first order divided differences.
%   Make three copies for the three approximations
%   (or four, if all four possible approximations are desired).
if(approx4)
  dL = cell(4,1);
  dR = cell(4,1);
else
  dL = cell(3,1);
  dR = cell(3,1);
end

% Take leftmost grid.N(dim) entries for left approximation.
indices1{dim} = 1 : size(D1, dim) - 1;
[ dL{:} ] = deal(D1(indices1{:}));

% Take rightmost grid.N(dim) entries for right approximation.
indices1{dim} = 2 : size(D1, dim);
[ dR{:} ] = deal(D1(indices1{:}));

%---------------------------------------------------------------------------
% Each copy gets modified by one of the second order terms.
%   Second order terms are sorted left to right.
% We'll build the middle approximation by going left then right
%   So for second order, use the leftward D2 term (indices1).
% In the four approximation case, we'll do the other direction as well.

% Coefficients for second order depend only on left or right approximation
%   (from O&F, depends only on k = i-1 (left) or k = i (right)).
coeffL = +1 * grid.dx(dim);
coeffR = -1 * grid.dx(dim);

indices1{dim} = 1 : size(D2, dim) - 2;
indices2{dim} = 2 : size(D2, dim) - 1;
dL{1} = dL{1} + coeffL * D2(indices1{:});
dL{2} = dL{2} + coeffL * D2(indices1{:});
dL{3} = dL{3} + coeffL * D2(indices2{:});
if(approx4)
  dL{4} = dL{4} + coeffL * D2(indices2{:});
end

indices1{dim} = indices1{dim} + 1;
indices2{dim} = indices2{dim} + 1;
dR{1} = dR{1} + coeffR * D2(indices1{:});
dR{2} = dR{2} + coeffR * D2(indices1{:});
dR{3} = dR{3} + coeffR * D2(indices2{:});
if(approx4)
  dR{4} = dR{4} + coeffR * D2(indices2{:});
end

%---------------------------------------------------------------------------
% Each copy gets modified by one of the third order terms.
%   Third order terms are sorted left to right.
% We'll build the middle approximation by going left then right.
%   So for the third order, use the rightward D3 term (indices2).
% In the four approximation case, we'll do the other direction as well.

% Coefficients for third order depend on second order term chosen
%   (from O&F, depends on k* = k-1 (left choice) or k* = k (right choice)).
% The second L or R refers to whether we went left or right on the D2 term.
coeffLL = +2 * grid.dx(dim)^2;
coeffLR = -1 * grid.dx(dim)^2;
coeffRL = -1 * grid.dx(dim)^2;
coeffRR = +2 * grid.dx(dim)^2;

indices1{dim} = 1 : size(D3, dim) - 3;
dL{1} = dL{1} + coeffLL * D3(indices1{:});
indices1{dim} = indices1{dim} + 1;
dL{2} = dL{2} + coeffLL * D3(indices1{:});
if(approx4)
  dL{4} = dL{4} + coeffLR * D3(indices1{:});
end
indices1{dim} = indices1{dim} + 1;
dL{3} = dL{3} + coeffLR * D3(indices1{:});

indices1{dim} = 2 : size(D3, dim) - 2;
dR{1} = dR{1} + coeffRL * D3(indices1{:});
indices1{dim} = indices1{dim} + 1;
dR{2} = dR{2} + coeffRL * D3(indices1{:});
if(approx4)
  dR{4} = dR{4} + coeffRR * D3(indices1{:});
end
indices1{dim} = indices1{dim} + 1;
dR{3} = dR{3} + coeffRR * D3(indices1{:});
