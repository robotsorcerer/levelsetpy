function [ derivL, derivR ] = upwindFirstENO3(grid, data, dim, generateAll)
% upwindFirstENO3: third order upwind approx of first derivative.
%
%   [ derivL, derivR ] = upwindFirstENO3(grid, data, dim, generateAll)
%
% Computes a third order directional approximation to the first
%   derivative, using an Essentially Non-Oscillatory (ENO) approximation.
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

if(nargin < 4)
  generateAll = 0;
end

[ derivL, derivR ] = upwindFirstENO3a(grid, data, dim, generateAll);
