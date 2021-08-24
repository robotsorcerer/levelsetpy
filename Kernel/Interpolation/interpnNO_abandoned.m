function values = interpnNO(grid, data, locations, method, extrap_val)
% interpnNO: Non-oscillatory interpolation of function values.
%
%   values = interpnNO(grid, data, locations, method, extrap_val)
%
% Interpolates function values from a grid in arbitrary dimension using
% Essentially Non-Oscillatory (ENO) or Weighted Essentially Non-Oscillatory
% (WENO) schemes.  For a good survey of these schemes, see Shu,
% "Essentially Non-Oscillatory and Weighted Essentially Non-Oscillatory
% Schemes for Hyperbolic Conservation Laws" in Advanced Numerical
% Approximation of Nonlinear Hyperbolic Equations (Cockburn, Johnson, Shu &
% Tadmor, eds), Lecture Notes in Mathematics 1697, pp. 325 - 432 (Springer:
% 1998).  Note that here we are interpolating function value from point
% samples of the function at the grid nodes, which is slightly different
% from either of the cases covered in that text.  The ENO3, ENO4, WENO4 and
% WENO6 schemes correspond to the ENO2, ENO3, WENO3 and WENO5 schemes from
% that survey (which are standard in the literature).
%
% In dimensions higher than one, interpolation proceeds in a dimension by
% dimension fashion (which can become extremely expensive in higher
% dimensions).
%
% Input Parameters:
%
%   grid: Grid structure (see processGrid.m for details).  The grid nodes
%   are used as abscissae in the interpolation.  A full grid structure is
%   required because it provides boundary conditions for populating ghost
%   cells and it ensures that the grid is equally spaced (which is required
%   by the ENO/WENO interpolation algorithms).
%
%   data: Double array.  The function values at the grid nodes, over which
%   the interpolation will be performed.
%
%   locations: Cell vector of length grid.dim.  Each element of the cell
%   array is a regular array, and all of these regular arrays must be the
%   same size.  Specifies the locations where the interpolated values
%   should be computed.  The regular array in cell entry i gives the
%   coordinates in dimension i.
%
%   method: String.  Specifies which interpolation scheme to use.
%   Optional.  The options are:
%
%     nearest: Nearest neighbor "interpolation" (first order accurate).
%     Implemented through a call to Matlab's standard interpn() routine.
%
%     linear: (Multi-)Linear interpolation (second order accurate).
%     Implemented through a call to Matlab's standard interpn() routine.
%
%     ENO3: ENO interpolation choosing between two third order accurate
%     parabolic interpolants.  Default.
%
%     ENO4: ENO interpolation choosing between three fourth order accurate
%     cubic interpolants.
%
%     WENO4: WENO interpolation blending two third order accurate parabolic
%     interpolants, which will be fourth order accurate in smooth regions.
%
%     WENO6: WENO interpolation blending three fourth order accurate cubic
%     interpolants, which will be sixth order accurate in smooth regions.
%
%   extrap_val: Scalar.  Value to return when an interpolation point falls
%   outside of the grid (eg: extrapolation was attempted).  Optional.
%   Default is NaN.  Note that this is not the same behaviour as Matlab's
%   interpn() (which silently extrapolates).
%
% Output Parameters:
%
%     values: Double array, the same size as the arrays in the cell entries
%     of locations.  The interpolated value of the function at each
%     location.

% Copyright 2010 Ian M. Mitchell (mitchell@cs.ubc.ca).
% This software is used, copied and distributed under the licensing 
%   agreement contained in the file LICENSE in the top directory of 
%   the distribution.
%
% Ian Mitchell, 8/09/2010
% $Date: 2010-08-30 01:43:11 -0700 (Mon, 30 Aug 2010) $
% $Id: interpnNO_abandoned.m 56 2010-08-30 08:43:11Z mitchell $

  %---------------------------------------------------------------------------
  %% Set optional parameters.
  if(nargin < 4)
    method = 'eno3';
  else
    method = lower(method);
  end

  if(nargin < 5)
    extrap_val = NaN;
  end
  
  %---------------------------------------------------------------------------
  %% For 'linear' or 'nearest' methods, call Matlab's interpn().
  if(strcmp(method, 'linear') || strcmp(method, 'nearest'))
    values = interpn(grid.xs{:}, data, locations{:}, method);    
    % Interpolation is completed, so skip the rest of the file.
    return
  end

  % If we reach this point then we are planning to do ENO/WENO interpolation.
  %---------------------------------------------------------------------------
  %% Figure out where we need to build interpolants.
  % Create a mask for those locations which are outside the grid
  % ("invalid").  Extract the locations inside the grid ("valid") into a
  % vector.  Determine the indices of the grid node in the lower corner of
  % the cells containing valid interpolation locations.

  % All of the location{} arrays should be the same size, otherwise an
  % error will be generated.
  output_size = size(locations{1});
  for i = 2 : grid.dim
    if (len(output_size) ~= len(size(locations{i})) || ...
        any(output_size ~= size(locations{i})))
      error('Entries of location cell vector must be the same size');
    end
  end

  % Determine a validity mask.
  valid_mask = ones(output_size);
  for i = 1 : grid.dim
    valid_mask = (valid_mask & (locations{i} >= grid.min(i)) ...
                  & (locations{i} <= grid.max(i)));
  end

  % Convert to an index set for valid entries.
  valid_indices = find(valid_mask);
  num_valid = length(valid_indices);

  % Extract valid locations (in the process turning them into a vector) and
  % determine indices of grid node in the lower left corner of the cell
  % containing those locations.
  corner_index = cell(grid.dim, 1);
  valid_loc = cell(grid.dim, 1);
  for i = 1 : grid.dim
    valid_loc{i} = locations{i}(valid_indices);
    corner_index{i} = floor((valid_loc{i} - grid.min(i)) * (1 / grid.dx(i)));
  end
  
  %---------------------------------------------------------------------------
  %% Determine stencil size.
  % We measure the number of samples needed on each side, in addition to
  % the two samples on the cell in which the interpolation location resides
  % (eg: the two samples which would be used for linear interpolation).
  switch(lower(method))
    case { 'eno3', 'weno4' }
      stencil_half = 1;
    case { 'eno4', 'weno6' }
      stencil_half = 2;
    otherwise
      error('Unknown method: %s', method)
  end
  
  % Total number of samples needed.
  stencil_full = 2 * stencil_half + 2;

  % Number of levels required in the divided difference table for the
  % interpolants.
  divided_difference_levels = stencil_half + 1;
  
  %---------------------------------------------------------------------------
  if(dim == 1)
    %% One dimensional grids are a special case.
    % One dimensional grids need no intermediate interpolants.
  

  %---------------------------------------------------------------------------
  else
    %% Determine locations of intermediate (ntrmdt) interpolants.
    % For grids of dimension greater than one, interpolation proceeds in a
    % dimension by dimension fashion (the ENO / WENO versions of multi-linear
    % interpolation).  We need to determine the locations on cell edges /
    % faces where these intermediate interpolants must be constructed.
    %
    % We are doing something very similar to ndgrid(), except each output
    % array of ndgrid() differs in only one dimension, while each output
    % array in this case will differ in two dimensions: the first dimension
    % (representing the actual interpolation locations) and the i+1
    % dimension (representing the intermediate interpolation locations in
    % dimension i).
    ntrmdt_loc = cell(grid.dim - 1, 1);

    for i = 1 : grid.dim - 1
      % I could not figure out any way to vectorize construction of the
      % columns of this index array.

      % Create an array which has more than 1 element in dimensions 1 and i.
      ntrmdt_temp = zeros([ num_valid, ones(1, i - 1), ...
                            stencil_full, ones(1, grid.dim - i - 1) ]);
      for j = -stencil_half : stencil_half + 1
        % Fill that array with the stencil elements.
        ntrmdt_temp = cat(i, ntrmdt_temp, corner_index{i} + j);

        NEED TO DO SOMETHING ABOUT BOUNDARY CONDITIONS
      end

      % Replicate that array in the remaining dimensions.
      ntrmdt_loc{i} = repmat(ntrmdt_temp, [ 1, stencil_full * ones(1, i - 1), ...
                                            1, stencil_full * ones(1, grid.dim - i - 1) ]);
    end
  
    %---------------------------------------------------------------------------
    %% Interpolate in the final dimension.
    % This computation is not terribly efficient, since we build the divided
    % difference table for the entire data array even if we are only
    % interpolating one point.  It would be better to do the divided
    % difference for the subset of the data relevant to the (intermediate)
    % interpolation locations, but that would require a more sophisticated,
    % indexed version of dividedDifferenceTable().
    dd_table = dividedDifferenceTable(grid, data, grid.dim, divided_difference_levels);
    
    % We construct Newton form polynomial interpolants.  The ENO / WENO
    % idea is to choose among the neighbour nodes to add to the stencil
    % based on which node is associated to the divided difference term of
    % minimum magnitude.

    % The location of each interpolant in the dimension of interest, copied
    % for all intermediate interpolation points.
    current_loc = repmat(valid_loc{grid.dim}, [ 1, stencil_full * ones(1, grid.dim - 1) ]);
    
    % The constant (degree zero) term is always the node to the left.
    ntrmdt_values = data(ntrmdt_loc{:});
    
    % The linear (degree one) term always adds the node to the right and
    % uses the associated first divided difference.  Note that the basis
    % function for the degree one Newton polynomial will be (location -
    % <node to the left>).
    newton_basis = current_loc - grid.xs{grid.dim}(ntrmdt_loc{:});
    ntrmdt_values = ntrmdt_values + dd_table{1}(ntrmdt_loc{:}) .* newton_basis);
    
    % Remaining terms (degree two and higher) add nodes which pick the
    % smallest corresponding entry in the divided difference table.
    eno_loc = ntrmdt_loc;
    for i = 2 : divided_difference_levels

      % Get the next basis function, which depends on the last node added.
      newton_basis = newton_basis .* (current_loc - grid.xs{grid.dim}(
      
      % Get indices for the two choices of entries in the divided
      % difference table.
      left_loc = eno_loc;
      right_loc = left_loc;
      right_loc{grid.dim} = right_loc{grid.dim} + 1;

      % Pick the entry of minimum magnitude.
      right_smaller = abs(dd_table{i}(left_loc)) > abs(dd_table{i}(right_loc));
      eno_loc{grid.dim} = left_loc{grid.dim} + right_smaller;
      
      THERE ARE SOME MAJOR INDEXING ISSUES WITH THIS APPROACH!!!!!!!!!!            
    
  end % Code for grids of dimension greater than one.
  %---------------------------------------------------------------------------
  %% Create the returned array.
  % Set all return values as if they were invalid.
  values = extrap_val * ones(output_size);

  % Fill in the valid values.
  values(valid_indices) = interp_results;
  
end

