function [ curvature, gradMag ] = curvatureSecond(grid, data)
% curvatureSecond: second order centered difference approx of the curvature.
%
%   [ curvature, gradMag ] = curvatureSecond(grid, data)
%
% Computes a second order centered difference approximation to the curvature.
%
%       \kappa = divergence(\grad \phi / | \grad \phi |)
%
% See O&F section 1.4 for more details.  In particular, this routine
%   implements equation 1.8 for calculating \kappa.
%
% parameters:
%   grid	Grid structure (see processGrid.m for details).
%   data        Data array.
%
%   curvature   Curvature approximation (same size as data).
%   gradMag	Magnitude of gradient |\grad \phi|
%                 Incidentally calculated while finding curvature,
%                 also second order centered difference.

% Copyright 2004 Ian M. Mitchell (mitchell@cs.ubc.ca).
% This software is used, copied and distributed under the licensing 
%   agreement contained in the file LICENSE in the top directory of 
%   the distribution.
%
% Ian Mitchell, 6/3/03

%---------------------------------------------------------------------------
% Get the first and second derivative terms.
[ second, first ] = hessianSecond(grid, data);

%---------------------------------------------------------------------------
% Compute gradient magnitude.
gradMag2 = first{1}.^2;
for i = 2 : grid.dim
  gradMag2 = gradMag2 + first{i}.^2;
end
gradMag = sqrt(gradMag2);

%---------------------------------------------------------------------------
curvature = zeros(size(data));
for i = 1 : grid.dim;
  curvature = curvature + second{i,i} .* (gradMag2 - first{i}.^2);
  for j = 1 : i - 1
    curvature = curvature - 2 * first{i} .* first{j} .* second{i,j};
  end
end

% Be careful not to stir the wrath of "Divide by Zero".
%  Note that gradMag == 0 implies curvature == 0 already, since all the
%  terms in the curvature approximation involve at least one first dervative.
nonzero = find(gradMag > 0);
curvature(nonzero) = curvature(nonzero) ./ gradMag(nonzero).^3;
