function data = shapeHyperplaneByPoints(grid, points, positivePoint)
% shapeHyperplaneByPoints: implicit surface function for a hyperplane.
%
%   data = shapeHyperplaneByPoints(grid, points, positivePoint)
%
% Creates a signed distance function for a hyperplane.  Unlike
% shapeHyperplane, this version accepts a list of grid.dim points which lie
% on the hyperplane.
%
% The direction of the normal (which determines which side of the hyperplane
% has positive values) is determined by one of two methods:
%
%   1) If the parameter positivePoint is provided, then the normal
%   direction is chosen so that the value at this point is positive.
%
%   2) If the parameter positivePoint is not provided, then it is assumed
%   that the points defining the hyperplane are given in "clockwise" order
%   if the normal points out of the "clock".  This method does not work
%   in 2D.
%
% Input Parameters:
%
%   grid: Grid structure (see processGrid.m for details).
%
%   points: Matrix specifying the points through which the hyperplane should
%   pass.  Each row is one point.  This matrix must be square of dimension
%   grid.dim.
%
%   positivePoint: Vector of length grid.dim specifying a point which lies
%   on the positive side of the interface.  This point should be within the
%   bounds of the grid.  Optional.  The method for determining the normal
%   direction to the hyperplane depends on whether this parameter is
%   supplied; see the discussion above for more details.  It is an error if
%   this point lies on the hyperplane defined by the other points.
%
% Output Parameters:
%
%   data: Output data array (of size grid.size) containing the implicit
%   surface function for the hyperplane.

% Copyright 2007 Ian M. Mitchell (mitchell@cs.ubc.ca).
% This software is used, copied and distributed under the licensing 
%   agreement contained in the file LICENSE in the top directory of 
%   the distribution.
%
% Ian Mitchell, 3/29/05
% Modified to add positivePoint option, Ian Mitchell 5/26/07
% $Date: 2009-09-03 16:34:07 -0700 (Thu, 03 Sep 2009) $
% $Id: shapeHyperplaneByPoints.m 44 2009-09-03 23:34:07Z mitchell $
s
%---------------------------------------------------------------------------
% For the positivePoint parameter, what is "too close" to the interface?
small = 1e3 * eps;

%---------------------------------------------------------------------------
if(nargin < 3)
  check_positive_point = 0;
else
  check_positive_point = 1;
end

%---------------------------------------------------------------------------
% Check that we have the correct number of points,
%   and they are linearly independent.
if(any(size(points) ~= [ grid.dim, grid.dim ]))
  error('Number of points must be equal to grid dimension');
end

%---------------------------------------------------------------------------
% We single out the first point.  Lines from this point to all the others
% should lie on the hyperplane.
point0 = points(1,:);
A = points(2:end,:) - repmat(point0, grid.dim - 1, 1);

% Extract the normal to the hyperplane.
normal = null(A);

% Check to see that it is well defined.
if(size(normal, 2) ~= 1)
  error('There does not exist a unique hyperplane through these points');
end

%---------------------------------------------------------------------------
% Signed distance function calculation.
%   This operation is just phi = n^T (x - p), but over the entire grid of x.
data = cellMatrixMultiply(num2cell(normal'), ...
                          cellMatrixAdd(grid.xs, num2cell(-point0')));

% In fact, the cellMatrix operation generates a 1x1 cell matrix
%   whose contents are the data array.
data = data{1};

%---------------------------------------------------------------------------
% The procedure above generates a correct normal assuming that the data
% points are given in a clockwise fashion.  If the user supplies
% parameter positivePoint, we need to use a different test.
if check_positive_point
  positivePointCell = num2cell(positivePoint);
  positivePointValue = interpn(grid.xs{:}, data, positivePointCell{:});
  if isnan(positivePointValue)
    error('positivePoint must be within the bounds of the grid.');
  elseif(abs(positivePointValue) < small)
    error('positivePoint parameter is too close to the hyperplane.');
  elseif(positivePointValue < 0)
    data = -data;
  end
end

%---------------------------------------------------------------------------
% Warn the user if there is no sign change on the grid
%  (ie there will be no implicit surface to visualize).
if(all(data(:) < 0) || (all(data(:) > 0)))
  warning([ 'Implicit surface not visible because function has ' ...
            'single sign on grid' ]);
end
