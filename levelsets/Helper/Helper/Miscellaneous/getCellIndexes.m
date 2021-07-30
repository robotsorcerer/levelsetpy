function [ cell_indexes, valid_mask ] = getCellIndexes(grid, locations)
% getCellIndexes: Determine within which grid cell(s) sample point(s) lie.
%
%   [ cell_indexes, valid_mask ] = getCellIndexes(grid, locations)
%
% Points which do not lie on a grid node will lie in a grid cell, which is
% the (hyper-)rectangle between grid nodes.  If we identify each grid cell
% with the node lying in its lower left corner (the node adjacent to the
% grid cell with smallest subscript index in every dimension), then we can
% label each grid cell.  This routine determines that lower left node (and
% hence the grid cell) for every point in the locations cell vector.
%
% Mathematically, a grid cell is the product of a collection of intervals,
% each of which is closed on its lower end and open on its upper end.
% Consequently, a grid cell only contains the node in the lower left corner
% (all other adjacent nodes are contained in a different grid cell).
%
% Input parameters:
%
%   grid: Grid structure (see processGrid.m for details).
%
%   locations: Cell vector.  Each element contains an array (all the same
%   size).  The entries in element i contain the location in dimension i of
%   the points.
%
% Output parameters:
%
%   cell_indexes: Array the same size as the elements of locations.  Linear
%   indexes of the nodes in the lower left corner of the cells
%   corresponding to the points in the location array.  For locations
%   outside the grid, NaN is returned.  These linear indexes can be
%   converted to subscript indexes by ind2sub(grid.shape, cell_indexes).
%
%   valid_mask: Boolean array the same size as cell_indexes.  Identifies
%   those points in the locations array which were within the grid.
%
% Ian Mitchell, 2012/08/09

  % Determine which locations are within the grid.
  location_size = size(locations{1});
  valid_mask = true(location_size);
  for i = 1 : grid.dim
    valid_mask = valid_mask & (locations{i} >= grid.min(i)) & (locations{i} <= grid.max(i));
  end
  
  % Determine the subscript indexes of the valid locations.
  cell_subscripts = cell(grid.dim, 1);
  for i = 1 : grid.dim
    cell_subscripts{i} = floor((locations{i}(valid_mask) - grid.min(i)) / grid.dx(i)) + 1;
  end

  % Set the return array to be NaN so that the invalid locations are
  % correctly flagged.
  cell_indexes = nan(location_size);
  % Determine the linear indexes of the valid locations.
  cell_indexes(valid_mask) = sub2ind(grid.shape, cell_subscripts{:});
  
end

