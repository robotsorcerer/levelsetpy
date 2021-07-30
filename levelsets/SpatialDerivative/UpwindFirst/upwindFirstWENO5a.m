function [ derivL, derivR ] = upwindFirstWENO5a(grid, data, dim, generateAll)
% upwindFirstWENO5a: fifth order upwind approx of first deriv by divided diffs.
%
%   [ derivL, derivR ] = upwindFirstWENO5a(grid, data, dim, generateAll)
%
% Computes a fifth order directional approximation to the first derivative, 
%   using a Weighted Essentially Non-Oscillatory (WENO) approximation.
%
% The ENO approximations are constructed by a divided difference table,
%   which is more efficient (although a little more complicated)
%   than using the direct equations from O&F section 3.4
%   (see upwindFirstWENO5b for that version).
%
% The smoothness estimates are computed from the first divided difference
%   table.  The left and right estimates are computed together,
%   taking advantage of the symmetries in the equations.  The results
%   should be the same as (3.32) - (3.34) in section 3.4 of O&F.
%
% The generateAll option is used for debugging, and possibly by
%   higher order weighting schemes.  Under normal circumstances
%   the default (generateAll = false) should be used.  Notice that
%   the generateAll option will just return the three ENO approximations.
%
% parameters:
%   grid	Grid structure (see processGrid.m for details).
%   data        Data array.
%   dim         Which dimension to compute derivative on.
%   generateAll Return all possible third order upwind approximations.
%                 If this boolean is true, then derivL and derivR will
%                 be cell vectors containing all the approximations
%                 instead of just the WENO approximation.  Note that
%                 the ordering of these approximations may not be 
%                 consistent between upwindFirstWENO1 and upwindFirstWENO2.
%                 (optional, default = 0)
%
%   derivL      Left approximation of first derivative (same size as data).
%   derivR      Right approximation of first derivative (same size as data).

% Copyright 2004 Ian M. Mitchell (mitchell@cs.ubc.ca).
% This software is used, copied and distributed under the licensing 
%   agreement contained in the file LICENSE in the top directory of 
%   the distribution.
%
% Ian Mitchell, 1/26/04

%---------------------------------------------------------------------------
if((dim < 0) | (dim > grid.dim))
  error('Illegal dim parameter');
end

if(nargin < 4)
  generateAll = 0;
end

% How would you like to calculate epsilon?
%   'constant'         Use constant 1e-6.
%   'maxOverGrid'      Scale by maximum D1.^2 term over the entire grid.
%   'maxOverNeighbors' Scale by maximum D1.^2 term among five neighbors.
% Compared to 'constant' (fastest), 'maxOverGrid' is about ~3% slower
%   and 'maxOverNeighbors' about ~17% slower.
% 'maxOverNeighbors' is the recommended method in O&F equation (3.38).
epsilonCalculationMethod = 'constant';
epsilonCalculationMethod = 'maxOverGrid';
%epsilonCalculationMethod = 'maxOverNeighbors';

%---------------------------------------------------------------------------
if(generateAll)

  % We only need the three ENO approximations 
  [ derivL, derivR ] = upwindFirstENO3aHelper(grid, data, dim, 0);
    
%---------------------------------------------------------------------------
else

  % We need the three ENO approximations 
  %   plus the (unstripped) divided differences to pick the least oscillatory.
  [ dL, dR, DD ] = upwindFirstENO3aHelper(grid, data, dim, 0, 0);

  % The smoothness estimates may have some relation to the higher order
  %   divided differences, but it isn't obvious from just reading O&F.
  % For now, use only the first order divided differences.
  D1 = DD{1};
  
  %---------------------------------------------------------------------------
  % Create cell array with array indices.
  sizeData = size(data);
  indices1 = cell(grid.dim, 1);
  for i = 1 : grid.dim
    indices1{i} = 1:sizeData(i);
  end
  indices2 = indices1;

  terms = 5;
  indices = cell(terms, 1);
  [ indices{:} ] = deal(indices1);

  % Element i of the indices cell vector contains an index cell vector
  %   that pulls out the v_i terms for the left approximation from the
  %   first divided difference table.
  for i = 1 : terms
    indices{i}{dim} = i : size(D1, dim) + i - 5;
  end
  
  %---------------------------------------------------------------------------
  % Smoothness estimates.
  smooth = cell(3,1);
  smooth{1} = ((13/12) * (D1(indices{1}{:}) ...
                          - 2 * D1(indices{2}{:}) ...
                          + D1(indices{3}{:})) .^2 ...
               + (1/4) * (D1(indices{1}{:}) ...
                          - 4 * D1(indices{2}{:}) ...
                          + 3 * D1(indices{3}{:})) .^2);
  smooth{2} = ((13/12) * (D1(indices{2}{:}) ...
                          - 2 * D1(indices{3}{:}) ...
                          + D1(indices{4}{:})) .^2 ...
               + (1/4) * (D1(indices{2}{:}) ...
                          - D1(indices{4}{:})) .^2);
  smooth{3} = ((13/12) * (D1(indices{3}{:}) ...
                          - 2 * D1(indices{4}{:}) ...
                          + D1(indices{5}{:})) .^2 ...
               + (1/4) * (3 * D1(indices{3}{:}) ...
                          - 4 * D1(indices{4}{:}) ...
                          + D1(indices{5}{:})) .^ 2);

  % Left smoothness estimates just use the left looking portion of
  %   these estimates.  The ENO approximations are in the same order
  %   as in O&F, so we can use the same alpha weights as (3.35) - (3.37).
  smoothL = cell(size(smooth));
  indices1{dim} = 1 : size(data, dim);
  for i = 1 : length(smooth)
    smoothL{i} = smooth{i}(indices1{:});
  end
  weightL = [ 0.1; 0.6; 0.3 ];
  
  % Right smoothness estimates are the same, but with D1 in the opposite order.
  %   Fortunately, the estimates are symmetric if we swap v1 for v5, 
  %   v2 for v4, and take the right looking portion of the estimates.
  % Note that the ENO approximations (and smoothness estimates)
  %   are in the opposite order as O&F, so we need to reorder the alpha
  %   weights from (3.35) - (3.37).
  smoothR = cell(size(smooth));
  indices2{dim} = 2 : size(data, dim) + 1;
  for i = 1 : length(smooth)
    smoothR{i} = smooth{i}(indices2{:});
  end
  weightR = [ 0.3; 0.6; 0.1 ];
  
  %---------------------------------------------------------------------------
  switch(epsilonCalculationMethod)
   case 'constant'
    epsilonL = 1e-6;
    epsilonR = epsilonL;
   case 'maxOverGrid'
    D1squared = D1.^2;
    epsilonL = 1e-6 * max(D1squared(:)) + 1e-99;
    epsilonR = epsilonL;
   case 'maxOverNeighbors'
    % Implements (3.38) in O&F for computing epsilon.
    D1squared = D1.^2;
    epsilon = D1squared(indices{1}{:});
    for i = 2 : length(indices)
      epsilon = max(epsilon, D1squared(indices{i}{:}));
    end
    epsilon = 1e-6 * epsilon + 1e-99;
    epsilonL = epsilon(indices1{:});
    epsilonR = epsilon(indices2{:});
   otherwise
    error('Unknown epsilonCalculationMethod %s', epsilonCalculationMethod);
  end
  
  %---------------------------------------------------------------------------
  % Compute and apply weights to generate a higher order WENO approximation.
  derivL = weightWENO(dL, smoothL, weightL, epsilonL);
  derivR = weightWENO(dR, smoothR, weightR, epsilonR);

end



%---------------------------------------------------------------------------
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%---------------------------------------------------------------------------
function deriv = weightWENO(d, s, w, epsilon)
% deriv = weightWENO(d, s, w, epsilon)
%
% Helper function to compute and apply WENO weighting terms.

% Compute weighting terms
alpha1 = w(1) ./ (s{1} + epsilon).^2;
alpha2 = w(2) ./ (s{2} + epsilon).^2;
alpha3 = w(3) ./ (s{3} + epsilon).^2;
sum = (alpha1 + alpha2 + alpha3);

deriv = (alpha1 .* d{1} + alpha2 .* d{2} + alpha3 .* d{3}) ./ sum;
