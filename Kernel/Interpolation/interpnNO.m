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
%     nearest, linear, spline, or cubic: Call's Matlab's standard interpn()
%     routine (except for 1D grids, in which case interp1() is called).
%
%     ENO1: Constant interpolation.  Always uses the value of the node to the
%     "left" (in all dimensions), so this choice is not the same as 'nearest'
%     and will in practice have a larger error than that option.  This option is
%     not recommended, but is included for completeness.
%
%     ENO2: (Multi-)Linear interpolation.  Effectively the same as Matlab's
%     multi-linear interpolation, but using the ENO code.  Note that there is
%     only one second order accurate linear interpolant, so there is actually no
%     choice involved in this interpolant.  This option is included for
%     completeness.
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
% $Date: 2012-01-03 16:13:54 -0800 (Tue, 03 Jan 2012) $
% $Id: interpnNO.m 72 2012-01-04 00:13:54Z mitchell $

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
  % Include extrap_val, since the extrapolation behaviour for this routine does
  % not quite agree with Matlab's.
  if(strcmp(method, 'linear') || strcmp(method, 'nearest') ...
     || strcmp(method, 'spline') || strcmp(method, 'cubic'))
    if(grid.dim == 1)
      % Stupid Matlab's stupid treatment of 1D arrays.
      values = interp1(grid.xs{:}, data, locations{:}, method, extrap_val);      
    else
      values = interpn(grid.xs{:}, data, locations{:}, method, extrap_val);
    end
    % Interpolation is completed, so skip the rest of the file.
    return
  end
  
  % If we reach this point then we are planning to do ENO/WENO interpolation.
  %---------------------------------------------------------------------------
  %% Figure out where we need to build interpolants.
  % Create a mask for those locations which are outside the grid ("invalid").
  % Extract the locations inside the grid ("valid") into a vector.  Determine
  % the indices of the grid node in the lower corner of the cells containing
  % valid interpolation locations.

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
  valid_mask = true(output_size);
  for i = 1 : grid.dim
    valid_mask = (valid_mask & (locations{i} >= grid.min(i)) ...
                  & (locations{i} <= grid.max(i)));
  end

  % Convert to an index set for valid entries.
  valid_indices = find(valid_mask);
  num_valid = length(valid_indices);
  
  % Extract valid locations (in the process turning them into column vectors)
  % and determine subscripts of grid node in the lower left corner of the cell
  % containing those locations.
  corner_sub = cell(grid.dim, 1);
  valid_loc = cell(grid.dim, 1);
  for i = 1 : grid.dim
    valid_loc{i} = locations{i}(valid_indices);
    corner_sub{i} = floor((valid_loc{i} - grid.min(i)) / grid.dx(i)) + 1;
  end
  
  %---------------------------------------------------------------------------
  %% Determine stencil nodes.
  % Offsets are measured from the node immediately to the left, so the node
  % immediately to the right is +1.
  switch(method)
    case 'eno1'
      stencil_offsets = 0;
    case 'eno2'
      stencil_offsets = [ 0, +1 ];
    case { 'eno3', 'weno4' }
      stencil_offsets = [ -1, 0, +1, +2 ];
    case { 'eno4', 'weno6' }
      stencil_offsets = [ -2, -1, 0, +1, +2, +3 ];
    otherwise
      error('Unknown method: %s', method)
  end
  
  % Total number of samples needed.
  stencil_size = length(stencil_offsets);

  %---------------------------------------------------------------------------
  
  
  %---------------------------------------------------------------------------
  if(grid.dim == 1)
    %% One dimensional grids are a special case.
    % One dimensional grids need no intermediate interpolants.

    % Find abscissae and data for the interpolation.  Some abscissae may be
    % outside the grid and should not be used for interpolation.  Fill the
    % abscissae (data_x) and data (data_y) arrays with NaN so that any
    % attempt to build an interpolant with these invalid data will be
    % detected.
    data_x = nan * ones(num_valid, stencil_size);
    data_y = nan * ones(size(data_x));
    data_invalid = false(size(data_x));
    for s = 1 : stencil_size
      % In 1D, subscripts and indexes are the same thing.
      data_index = corner_sub{1} + stencil_offsets(s);
      data_valid = ((data_index >= 1) & (data_index <= grid.N(1)));
      data_invalid(:,s) = ~data_valid;
      data_x(data_valid,s) = grid.vs{1}(data_index(data_valid));
      % Data array is one dimensional, so we can index into with just the
      % indexes for this dimension.
      data_y(data_valid,s) = data(data_index(data_valid));
    end
    sample_x = valid_loc{1};
    
    switch(method)
      case { 'eno1', 'eno2', 'eno3', 'eno4' }
        interp_results = interpHelperENO(data_x, data_y, sample_x, data_invalid);
      case { 'weno4', 'weno6' }
        error('Method not yet implemented: %s', method);
      otherwise
        error('Unknown method: %s', method);
    end

  %---------------------------------------------------------------------------
  elseif(grid.dim == 2)
    %% 2D tensor product code to see how it might be done.
    % Then we will try to abstract to arbitrary dimension.

    % In what order should be interpolate dimensions?
    interpolation_order = [ 2, 1 ];
    
    % Short variable names for the current and next dimension.
    dc = interpolation_order(1);
    dn = interpolation_order(2);
    
    % Construct abscissae and data for the second dimension.  Some
    % abscissae may be outside the grid and should not be used for
    % interpolation.  Fill the abscissae (data_x) and data (data_y) arrays
    % with NaN so that any attempt to build an interpolant with these
    % invalid data will be detected.

    data_x = nan * ones(num_valid, stencil_size^2);
    data_invalid = false(size(data_x));
    data_y = nan * ones(size(data_x));
    for sn = 1 : stencil_size
      % Loop over the blocks (which correspond to interpolation points in
      % the next dimension).
      block_rows = ((sn - 1) * num_valid) + (1 : num_valid);
      data_sub = corner_sub;
      data_sub{dn} = data_sub{dn} + stencil_offsets(sn);
      data_validn = ((data_sub{dn} >= 1) & (data_sub{dn} <= grid.N(dn)));

      for sc = 1 : stencil_size
        % Loop over the columns (which correspond to interpolation points
        % in the current dimension).
        data_sub{dc} = data_sub{dc} + stencil_offsets(sc);
        data_index = sub2ind(grid.shape, data_sub{:});
        data_validc = data_validn & ((data_sub{dc} >= 1) & (data_sub{dc} <= grid.N(dc)));
        data_x(block_rows(data_validc),sc) = grid.xs{dc}(data_index(data_validc));
        data_y(block_rows(data_validc),sc) = data(data_index(data_validc));
        data_invalid(block_rows,sc) = ~data_validc;
      end
    end
    
    % The sample locations are the same for each block.
    sample_x = repmat(valid_loc{dc}, stencil_size, 1);
    
    % Perform the interpolation.
    switch(method)
      case { 'eno1', 'eno2', 'eno3', 'eno4' }
        interp_results = interpHelperENO(data_x, data_y, sample_x, data_invalid);
      case { 'weno4', 'weno6' }
        error('Method not yet implemented: %s', method);
      otherwise
        error('Unknown method: %s', method);
    end

    % Reshape the results to work on the next dimension.
    
    
  %---------------------------------------------------------------------------
  else
    %% Grids of dimension greater than one.
    
    error('Interpolation not yet implemented for dimension %d', dim);
    
  end % Code for grids of dimension greater than one.
  %---------------------------------------------------------------------------
  %% Create the returned array.
  % Set all return values as if they were invalid.
  values = extrap_val * ones(output_size);

  % Fill in the valid values.
  values(valid_indices) = interp_results;
  
end % Main function.


%---------------------------------------------------------------------------
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%---------------------------------------------------------------------------
%function values = buildIntermediates(data, stencil_offsets, locations, indices, which_dim)
%
%

%end