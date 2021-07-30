function laplacian = laplacianSecond(grid, data)
% laplacian: second order centered difference approx of the Laplacian.
%
%   laplacian = laplacianSecond(grid, data)
%
% Computes a second order centered difference approximation to the Laplacian.
%
%       \Delta \phi = \grad \dot \grad \phi
%                   = \grad^2 \phi
%                   = sum_i d^2 \phi / d x_i^2
%
% parameters:
%   grid	Grid structure (see processGrid.m for details).
%   data        Data array.
%
%   laplacian   Laplacian approximation (same size as data).

% Copyright 2004 Ian M. Mitchell (mitchell@cs.ubc.ca).
% This software is used, copied and distributed under the licensing 
%   agreement contained in the file LICENSE in the top directory of 
%   the distribution.
%
% Ian Mitchell, 02/02/04

% Current implementation uses hessianSecond, which also computes the
%   second order mixed partial terms.
% If a good use is found for this routine, it would make sense to 
%   increase its efficiency by computing just the necessary second order terms.
  
%---------------------------------------------------------------------------
% Get the second derivative terms.
second = hessianSecond(grid, data);

%---------------------------------------------------------------------------
laplacian = second{1,1}
for i = 2 : grid.dim;
  laplacian = laplacian + second{i,i};
end
