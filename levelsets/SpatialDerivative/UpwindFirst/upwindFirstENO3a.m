function [ derivL, derivR ] = upwindFirstENO3a(grid, data, dim, generateAll)
% upwindFirstENO3a: third order upwind approx of first deriv by divided diffs.
%
%   [ derivL, derivR ] = upwindFirstENO3a(grid, data, dim, generateAll)
%
% Computes a third order directional approximation to the first
%   derivative, using an Essentially Non-Oscillatory (ENO) approximation.
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
% The generateAll option is used for debugging, and possibly by
%   higher order weighting schemes.  Under normal circumstances
%   the default (generateAll = false) should be used.
%
% parameters:
%   grid	Grid structure (see processGrid.m for details).
%   data        Data array.
%   dim         Which dimension to compute derivative on.
%   generateAll Return all possible third order upwind approximations.
%                 If this boolean is true, then derivL and derivR will
%                 be cell vectors containing all the approximations
%                 instead of just the ENO approximation.
%                 (optional, default = 0)
%
%   derivL      Left approximation of first derivative (same size as data).
%   derivR      Right approximation of first derivative (same size as data).

% Copyright 2004 Ian M. Mitchell (mitchell@cs.ubc.ca).
% This software is used, copied and distributed under the licensing 
%   agreement contained in the file LICENSE in the top directory of 
%   the distribution.
%
% Ian Mitchell, 1/23/04

%---------------------------------------------------------------------------
if((dim < 0) | (dim > grid.dim))
  error('Illegal dim parameter');
end

if(nargin < 4)
  generateAll = 0;
end

% Check that approximations that should be equivalent are equivalent 
%   (for debugging purposes, only used if generateAll == 1).
checkEquivalentApproximations = 1;
small = 100 * eps;             % a small number for "equivalence"

%---------------------------------------------------------------------------
if(generateAll)

  % We only need the three ENO approximations 
  %   (plus the fourth if we want to check for equivalence).
  [ dL, dR ] = upwindFirstENO3aHelper(grid, data, dim, ...
                                      checkEquivalentApproximations);
    
  %---------------------------------------------------------------------------
  if(checkEquivalentApproximations)
    % Only the LLL and RRR approximations are not equivalent to at least one
    %   other approximations, so we have several checks.

    % Check corresponding left and right approximations against one another.
    checkEquivalentApprox(dL{2}, dR{1}, small);
    checkEquivalentApprox(dL{3}, dR{2}, small);

    % Check the middle approximations.
    checkEquivalentApprox(dL{2}, dL{4}, small);
    checkEquivalentApprox(dR{2}, dR{4}, small);
    
  end

  %---------------------------------------------------------------------------
  % Caller requested all approximations in each direction.
  %   If we requested all four approximations above, strip off the last one.
  derivL = dL(1:3);
  derivR = dR(1:3);

%---------------------------------------------------------------------------
else

  % We need the three ENO approximations 
  %   plus the (stripped) divided differences to pick the least oscillatory.
  [ dL, dR, DD ] = upwindFirstENO3aHelper(grid, data, dim, 0, 1);
  
  %---------------------------------------------------------------------------
  % Create cell array with array indices.
  sizeData = size(data);
  indices1 = cell(grid.dim, 1);
  for i = 1 : grid.dim
    indices1{i} = 1:sizeData(i);
  end
  indices2 = indices1;

  %---------------------------------------------------------------------------
  % Need to figure out which approximation has the least oscillation.
  %   Note that L and R in this section refer to neighboring divided
  %   difference entries, not to left and right approximations.

  % Pick out minimum modulus neighboring D2 term.
  D2abs = abs(DD{2});
  indices1{dim} = 1 : size(D2abs, dim) - 1;
  indices2{dim} = indices1{dim} + 1;
  smallerL = (D2abs(indices1{:}) < D2abs(indices2{:}));
  smallerR = ~smallerL;

  %---------------------------------------------------------------------------
  % Figure out smallest modulus D3 terms, 
  %   given choice of smallest modulus D2 terms above.
  D3abs = abs(DD{3});
  indices1{dim} = 1 : size(D3abs, dim) - 1;
  indices2{dim} = indices1{dim} + 1;
  smallerTemp = (D3abs(indices1{:}) < D3abs(indices2{:}));

  indices1{dim} = 1 : size(smallerTemp, dim) - 1;
  indices2{dim} = indices1{dim} + 1;
  smallerLL = smallerTemp(indices1{:}) & smallerL;
  smallerRL = smallerTemp(indices2{:}) & smallerR;
  smallerTemp = ~smallerTemp;
  smallerLR = smallerTemp(indices1{:}) & smallerL;
  smallerRR = smallerTemp(indices2{:}) & smallerR;

  smallerM = smallerRL | smallerLR;
  
  %---------------------------------------------------------------------------
  % Pick out the best third order approximation
  indices1{dim} = 1 : size(smallerLL, dim) - 1;
  derivL = (dL{1} .* smallerLL(indices1{:}) ...
            + dL{2} .* smallerM(indices1{:})...
            + dL{3} .* smallerRR(indices1{:}));
  
  indices1{dim} = 2 : size(smallerLL, dim);
  derivR = (dR{1} .* smallerLL(indices1{:}) ...
            + dR{2} .* smallerM(indices1{:})...
            + dR{3} .* smallerRR(indices1{:}));

end
