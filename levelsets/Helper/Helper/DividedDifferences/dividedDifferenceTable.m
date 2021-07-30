function dd_table = dividedDifferenceTable(grid, data, dim, level, stripDD)
% dividedDifferenceTable: computes a divided difference table.
%
%   dd_table = dividedDifferenceTable(grid, data, dim, level, stripDD)
%
% Computes a standard divided difference table in the specified dimension
% for the specified data up to the specified level of divided differences.
% Ghost cells will be added to the data, and for levels higher than the
% first divided difference, this may result in low level divided difference
% entries which lie entirely within the ghost cells; by default these
% entries are stripped from the output.
%
% Input Parameters:
%
%   grid: Grid structure (see processGrid.m for details).  Used to
%   determine the dimension, boundary conditions and spacing for the
%   divided difference calculation.  Note that the grid node coordinates
%   are NOT used.
%
%   data: Double array.  The data on which the divided difference is to be
%   computed.
%
%   dim: Positive integer.  Which dimension to compute derivative on.
%
%   level: Positive integer.  What level of divided difference to
%   compute.  All lower level divided differences (down to the first
%   divided difference) will be included in the table.
%
%   stripDD: Boolean.  Strip the divided difference tables down to their
%   appropriate size, otherwise they will contain entries (at the lower
%   levels) that correspond entirely to ghost cells.  Optional.  Default = 1 
%   (eg: strip the ghost cell entries).
%
% Output Parameters:
%
%   dd_table: Cell vector containing the divided difference table.

% Copyright 2010 Ian M. Mitchell (mitchell@cs.ubc.ca).
% This software is used, copied and distributed under the licensing 
%   agreement contained in the file LICENSE in the top directory of 
%   the distribution.
%
% Ian Mitchell, 8/09/2010
% $Date: 2011-03-18 16:56:16 -0700 (Fri, 18 Mar 2011) $
% $Id: dividedDifferenceTable.m 60 2011-03-18 23:56:16Z mitchell $

  %---------------------------------------------------------------------------
  %% Set optional parameters.
  if(nargin < 5)
    stripDD = 1;
  end

  %---------------------------------------------------------------------------
  %% Set up the ghost cells.
  gdata = feval(grid.bdry{dim}, data, dim, level, grid.bdryData{dim});

  %---------------------------------------------------------------------------
  %% Create cell array with array indices.
  sizeData = size(gdata);
  indices1 = cell(grid.dim, 1);
  for i = 1 : grid.dim
    indices1{i} = 1:sizeData(i);
  end
  indices2 = indices1;

  %---------------------------------------------------------------------------
  %% Create cell array to hold the divided differences.
  dd_table = cell(level, 1);

  %---------------------------------------------------------------------------
  %% Compute (un)divided differences.
  last_level = gdata;
  for i = 1 : level
    indices1{dim} = 2 : size(last_level, dim);
    indices2{dim} = indices1{dim} - 1;
    dd_table{i} = last_level(indices1{:}) - last_level(indices2{:});
    last_level = dd_table{i};
  end

  %---------------------------------------------------------------------------
  %% Strip differences corresponding to ghost cells if necessary.
  if stripDD
    for i = 1 : level - 1
      indices1{dim} = 1 + (level - i) : size(dd_table{i}, dim) - (level - i);
      dd_table{i} = dd_table{i}(indices1{:});
    end
  end

  %---------------------------------------------------------------------------
  %% Perform the divisions to make it a divided difference table.
  % Implementation assumes a uniform grid in each dimension. To reduce
  % computational costs and improve accuracy, we compute the undivided
  % differences and only divide by grid spacing at the end.  The scalar
  % division below will cause an error if dx is not a scalar (eg: if the grid
  % is not uniformly spaced). If the decision to generalize to such grids is
  % made in the future, then the table computing code above will have to be
  % modified to use the divided differences.
  for i = 1 : level
    divisor_inv = 1 / (i * grid.dx(dim));
    dd_table{i} = divisor_inv * dd_table{i};
  end

end  % End of function.