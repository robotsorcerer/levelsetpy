function data = shapeHyperplane(grid, normal, point)
% shapeHyperplane: implicit surface function for a hyperplane.
%
%   data = shapeHyperplane(grid, normal, point)
%
% Creates a signed distance function for a hyperplane.
%
% Input Parameters:
%
%   grid: Grid structure (see processGrid.m for details).
%
%   normal:  Column vector specifying the outward normal of the hyperplane.
%
%   point: Vector specifying a point through which the hyperplane passes.
%   Defaults to the origin.
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
% $Id: shapeHyperplane.m 44 2009-09-03 23:34:07Z mitchell $

%---------------------------------------------------------------------------
% Default parameter values.
if(nargin < 3)
  point = zeros(grid.dim, 1);
end

%---------------------------------------------------------------------------
% Normalize the normal to be a unit vector.
normal = normal / norm(normal);

%---------------------------------------------------------------------------
% Signed distance function calculation.
%   This operation is just phi = n^T (x - p), but over the entire grid of x.
data = cellMatrixMultiply(num2cell(normal'), ...
                          cellMatrixAdd(grid.xs, num2cell(-point)));

% In fact, the cellMatrix operation generates a 1x1 cell matrix
%   whose contents are the data array.
data = data{1};

%---------------------------------------------------------------------------
% Warn the user if there is no sign change on the grid
%  (ie there will be no implicit surface to visualize).
if(all(data(:) < 0) || (all(data(:) > 0)))
  warning([ 'Implicit surface not visible because function has ' ...
            'single sign on grid' ]);
end
