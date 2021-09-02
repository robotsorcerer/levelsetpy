function data = shapeSphere(grid, center, radius)
% shapeSphere: implicit surface function for a sphere.
%
%   data = shapeSphere(grid, center, radius)
%
% Creates an implicit surface function (actually signed distance) 
%   for a sphere.
%
% Can be used to create circles in 2D or intervals in 1D.
%
%
% Input Parameters:
%
%   grid: Grid structure (see processGrid.m for details).
%
%   center: Vector specifying center of sphere.  May be a scalar, in
%   which case the scalar is multiplied by a vector of ones of the
%   appropriate length.  Defaults to 0 (eg centered at the origin).
%
%   radius: Scalar specifying the radius of the sphere. Defaults to 1.
%
% Output Parameters:
%
%   data: Output data array (of size grid.size) containing the implicit
%   surface function.

% Copyright 2004 Ian M. Mitchell (mitchell@cs.ubc.ca).
% This software is used, copied and distributed under the licensing 
%   agreement contained in the file LICENSE in the top directory of 
%   the distribution.
%
% Ian Mitchell, 6/23/04
% $Date: 2009-09-03 16:34:07 -0700 (Thu, 03 Sep 2009) $
% $Id: shapeSphere.m 44 2009-09-03 23:34:07Z mitchell $

%---------------------------------------------------------------------------
% Default parameter values.
if(nargin < 2)
  center = zeros(grid.dim, 1);
elseif(numel(center) == 1)
  center = center * ones(grid.dim, 1);
end

if(nargin < 3)
  radius = 1;
end

%---------------------------------------------------------------------------
% Signed distance function calculation.
data = (grid.xs{1} - center(1)).^2;
for i = 2 : grid.dim
  data = data + (grid.xs{i} - center(i)).^2;
end
data = sqrt(data) - radius;

%---------------------------------------------------------------------------
% Warn the user if there is no sign change on the grid
%  (ie there will be no implicit surface to visualize).
if(all(data(:) < 0) || (all(data(:) > 0)))
  warning([ 'Implicit surface not visible because function has ' ...
            'single sign on grid' ]);
end
