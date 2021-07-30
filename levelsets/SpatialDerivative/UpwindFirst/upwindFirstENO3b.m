function [ derivL, derivR ] = upwindFirstENO3b(grid, data, dim, generateAll)
% upwindFirstENO3b: third order upwind approx of first deriv by direct calc.
%
%   [ derivL, derivR ] = upwindFirstENO3b(grid, data, dim, generateAll)
%
% Computes a third order directional approximation to the first
%   derivative, using an Essentially Non-Oscillatory (ENO) approximation.
%
% The approximation is constructed by the equations in O&F, section 3.4
%   equations (3.25) - (3.27).  This is an 
%   alternative to the more efficient divided difference
%   table for computing the ENO approximations, which is used in
%   upwindFirstENO3a.  In particular, the left and right approximations
%   are computed independently in this version.
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
%                 instead of just the ENO approximation.  Note that
%                 the ordering of these approximations may not be 
%                 consistent between upwindFirstENO1 and upwindFirstENO2.
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

% How big is the stencil?
stencil = 3;

% Check that approximations that should be equivalent are equivalent 
%   (for debugging purposes, only used if generateAll == 1).
checkEquivalentApproximations = 1;
small = 100 * eps;             % a small number for "equivalence"

% Add ghost cells.
gdata = feval(grid.bdry{dim}, data, dim, stencil, grid.bdryData{dim});


%---------------------------------------------------------------------------
if(generateAll)

  % Compute the left and right approximations.
  % No need to build WENO approximation, just return all the ENO approx.
  derivL = upwindFirstENO3bHelper(grid, gdata, dim, -1);
  derivR = upwindFirstENO3bHelper(grid, gdata, dim, +1);
  
  %---------------------------------------------------------------------------
  % If necessary, check equivalence of ENO terms.  
  % Using notation of (3.25) - (3.27) for phi^i and the Left/Right 
  %   choices at the D^1, D^2 and D^3 levels in that order:
  %   For left approximation, phi^1 is LLL, phi^2 is LLR/LRL and phi^3 is LRR;
  %   For right approximation, phi^1 is RRR, phi^2 is RLR/RRL and phi^3 is RLL.
  % Hence the equivalences listed below.
  if(checkEquivalentApproximations)
    checkEquivalentApprox(derivL{2}, derivR{3}, small);
    checkEquivalentApprox(derivL{3}, derivR{2}, small);
  end
  
%---------------------------------------------------------------------------
else

  %---------------------------------------------------------------------------
  % Compute the left and right ENO approximations.
  [ dL, smoothL ] = upwindFirstENO3bHelper(grid, gdata, dim, -1);
  [ dR, smoothR ] = upwindFirstENO3bHelper(grid, gdata, dim, +1);
  
  %---------------------------------------------------------------------------
  % The best ENO approximant has the smallest smoothness estimate
  derivL = choose(dL, smoothL);
  derivR = choose(dR, smoothR);
  
end



%---------------------------------------------------------------------------
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%---------------------------------------------------------------------------
function deriv = choose(d, s)
% deriv = choose(d, s);
%
% Helper function to choose least oscillatory ENO approximant.


choose1over2 = (s{1} < s{2});
choose1over3 = (s{1} < s{3});
choose2over3 = (s{2} < s{3});

deriv = ((choose1over2 & choose1over3) .* d{1} ...
         + (~choose1over2 & choose2over3) .* d{2} ...
         + (~choose1over3 & ~choose2over3) .* d{3});

