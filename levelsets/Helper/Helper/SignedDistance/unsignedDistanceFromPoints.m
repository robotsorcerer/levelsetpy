function data = unsignedDistanceFromPoints(grid, points)
% unsignedDistanceFromPoints: Build distance function from surface points.
%
%   data = unsignedDistanceFromPoints(grid, points)
%
% Given a set of points lying on the surface of an object, this function
%   constructs an unsigned distance function to those points.  For each
%   point on the grid, the distance to every point in the point cloud is
%   computed, and the minimum is chosen.  While exact, this algorithm scales
%   as the product of the number of points in the cloud and the number of
%   grid nodes.  It should only be used for debugging and validation
%   purposes.  If this routine is being used seriously, it should be
%   rewritten to use some kind of binning in a kd-tree (ie quadtree in 2D
%   and octtree in 3D).  If an approximation is good enough, this routine 
%   should be replaced entirely by a fast marching method.
%
% Important notes: This algorithm will run VERY SLOWLY for large point
%   clouds or large grids.  It will not make any attempt to fill in holes
%   in the point cloud, so it should only be used to construct implicit
%   surface functions on grids coarse enough that there are several
%   points in the point cloud per grid cell.  Since it produces an
%   unsigned distance function, some other method will need to be used to
%   assign an inside and outside and thereby create the true implicit
%   surface function.
%
% Parameters:
%
%   grid	 Grid structure on which to build the signed distance function.
%   points       Collection of points on the surface, each point is a row
%                  vector of length grid.dim.
%
%   data         An unsigned distance function defined on all the grid nodes,
%                  giving the distance from that node to the nearest
%                  point in the point cloud.

% Copyright 2004 Ian M. Mitchell (mitchell@cs.ubc.ca).
% This software is used, copied and distributed under the licensing 
%   agreement contained in the file LICENSE in the top directory of 
%   the distribution.
%
% Ian Mitchell, 2/20/04
  
  if(grid.dim ~= size(points, 2))
    error('Number of columns of points argument must equal grid.dim');
  end
  
  pointsN = size(points, 1);

  % Construct (unsigned) distance function.
  %   We'll loop over the nodes, since there are often more data points
  %   than  nodes in the grid.
  data = inf * ones(grid.shape);
  node = zeros(1, grid.dim);
  for i = 1 : prod(grid.N);
    % Get vector for this node's state.
    for d = 1 : grid.dim
      node(d) = grid.xs{d}(i);
    end
    
    % Determine distance to nearest surface point.
    nodeListSurface = repmat(node, pointsN, 1);
    data(i) = min(sqrt(sum((points - nodeListSurface).^2, 2)));

  end
