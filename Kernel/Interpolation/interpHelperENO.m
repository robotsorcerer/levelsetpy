function sample_y = interpHelperENO(data_x, data_y, sample_x, data_invalid)
% interpHelperENO: Helper routine to construct ENO interpolants.
%
% Constructs one or more Essentially Non-Oscillatory (ENO) one dimensional
% interpolants of the same degree (which may be zero or higher), and evaluates
% each at a single point.  Because all interpolants are of the same degree, the
% potential stencil is the same width for all; however, the actual stencil is
% separately calculated for each according to ENO rules.
%
% This routine is designed as a helper for interpnNO, and hence its arguments
% are required in a rather specific and potentially inconvenient form.  Users
% should not generally call this routine, but rather call interpnNO instead.
%
% This routine is designed to work with equally spaced abscissae.  It could be
% relatively easily modified to handle unequally spaced abscissae; however, it
% would involve a little more code and require noticeably more division
% operations so it has not yet been done.
%
% Input Parameters:
%
%   data_x: 2D array.  Each row holds the abscissae for one interpolant.  The
%   number of columns must be 1 or even.  The stencil size and hence degree of
%   the interpolant will be deduced from the number of columns 
%
%       <degree of interpolant> = 0 if <number of columns> == 1
%                               = <number of columns> / 2 otherwise
%
%   The spacing of the abscissae (eg: dx) is also deduced from this array;
%   the abscissae must be equally spaced.
%
%   data_y: 2D array, same size as data_x with the same interpretation.
%   These are the data values corresponding to the abscissae.
%
%   sample_x: Column vector.  Each row gives the location at which the
%   corresponding interpolant will be evaluated.  It is assumed that in each row
%   the value of sample_x lies between the middle two abscissae in the
%   corresponding row of data_x; in other words, the stencil in data_x is
%   centered on the corresponding entry in sample_x.
%
%   data_invalid: 2D boolean array, same size as data_x.  Flags the entries in
%   data_y which may not be used in the actual stencils for the interpolants.
%   Divided difference table entries which arise from invalid entries in data_y
%   are never used in the interpolant construction.  It is assumed that there
%   are sufficient valid entries in each row that at least one interpolant of
%   full degree can be constructed.  Optional argument.  Default is that every
%   entry is valid.
%
% Output Parameters:
%
%   sample_y: Column vector.  Same size as sample_x.  The interpolated y values
%   at the x locations given by sample_x.

% Copyright 2010 Ian M. Mitchell (mitchell@cs.ubc.ca).
% This software is used, copied and distributed under the licensing 
%   agreement contained in the file LICENSE in the top directory of 
%   the distribution.
%
% Ian Mitchell, 8/22/2010
% $Date: 2011-06-29 22:10:51 -0700 (Wed, 29 Jun 2011) $
% $Id: interpHelperENO.m 70 2011-06-30 05:10:51Z mitchell $

  %---------------------------------------------------------------------------
  %% Optional arguments, measure and check input arguments for errors.
  % This routine is not supposed to be called by the user, but the bug-free
  % programmer's mantra is "trust no one".
  
  if(nargin < 4)
    data_invalid = zeros(size(data_y));
  end

  [ num_entries, stencil_full ] = size(data_x);

  if(stencil_full == 1)
    % Piecewise constant interpolation, which makes many of the other input
    % tests redundant.
    divided_difference_levels = 0;
    % Index of abscissa to the left of the interpolation location.
    middle_left = 1;
    % Set dx to be NaN in order to flag any attempt to incorrectly use dx.
    dx = nan;

  else
    % Stencil size must be even if it is not 1.
    if(rem(stencil_full, 2) ~= 0)
      error('Potential stencil must always have even width');
    end
    
    % Interpolation of degree greater than one.
    divided_difference_levels = stencil_full / 2;

    % Index of abscissa to the left of the interpolation location.
    middle_left = stencil_full / 2;
    % Index of abscissa to the right of the interpolation location.
    middle_right = middle_left + 1;

    if(any(sample_x < data_x(:, middle_left)) || any(sample_x > data_x(:,middle_right)))
      error('Interpolation location must lie between middle two abscissae in each stencil.');
    end
    
    % Check that dx is constant throughout the (valid) data.
    dx_all = diff(data_x, 1, 2);
    % The dx calculation is invalid if the data value to the left or right
    % is invalid.
    dx_valid = dx_all(~(data_invalid(:,1:end-1) | data_invalid(:,2:end)));
    dx = dx_valid(1);
    % We arbitrarily choose a cutoff of 10*epilson in relative and absolute
    % error as too large a difference in dx.
    if any(abs(dx_valid - dx) > (10 * dx * eps + 10 * eps))
      error('Abscissae spacing must be constant.');
    end
    
  end % stencil_full > 1 tests.
  
  if any(data_invalid(:, middle_left))
    error('Interpolant cannot be constructed if middle-left entry is invalid.');
  end

  %---------------------------------------------------------------------------
  %% Construct the divided difference table.
  % We are always taking the divided differences in dimension 2 (the columns)
  % because that dimension holds the various data values with which to build the
  % interpolants (the rows).
  dd_table = dividedDifferenceTableHelper(data_y, divided_difference_levels, dx);

  %---------------------------------------------------------------------------
  %% Construct ENO Newton basis interpolant.
  % We construct Newton form polynomial interpolants.  The ENO idea is to choose
  % among the neighbour nodes to add to the stencil based on which node is
  % associated to the divided difference term of minimum magnitude.

  % - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  %% Set up the iteration over levels of the divided difference table.
  % The first basis function is just constant.
  current_basis = 1;
  
  % We will need to build a lot of indexes into the data_x, data_y and
  % divided difference tables.  In each case, we will be picking one column
  % for each row.
  all_rows = (1 : num_entries)';
  
  % Because the stencil is centered, the node to the left of the sample location
  % for each row should always correspond to the same column in data_x and
  % data_y: the column just under halfway across.
  current_data_column = middle_left * ones(num_entries, 1);

  % The constant (degree zero) term in the interpolant is always the data value
  % of the node to the left.  Note that current_basis = 1 so it is not necessary
  % to include it in this calculation; however, it is included to be consistent
  % with the code inside the loop below.  We can ignore validity at this point
  % because the left of middle data values in the stencil were already confirmed
  % to be valid (otherwise the sample point would be outside the grid).
  data_y_index = sub2ind(size(data_y), all_rows, current_data_column);
  sample_y = data_y(data_y_index) .* current_basis;

  % The first abscissa is always the node to the left as well.
  current_abscissae_column = current_data_column;
  
  % The actual stencil is always a contiguous set of nodes.
  actual_stencil_lcolumn = current_abscissae_column;
  actual_stencil_rcolumn = current_abscissae_column;

  % - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  %% Iterate over the levels in the divided difference table.
  % The higher degree terms are taken from the divided difference table. 
  for i = 1 : divided_difference_levels
    
    % As we increase the degree of the interpolant, the Newton basis functions are
    % constructed using the abscissae that have come before.
    current_abscissae_index = sub2ind(size(data_x), all_rows, current_abscissae_column);
    current_basis = current_basis .* (sample_x - data_x(current_abscissae_index));

    if(i == 1)
      % The linear (degree one) term in the interpolant is always the divided
      % difference corresponding to the gap between the left and right nodes.
      % In effect, the right term is always treated as smaller for this level of
      % the divided difference table.
      left_smaller = zeros(num_entries, 1);

    else
      % For higher degree terms, determine the minimum magnitude divided
      % difference.
      dd_table_indexes_left = sub2ind(size(dd_table{i}), all_rows, current_data_column - 1);
      dd_table_indexes_right = sub2ind(size(dd_table{i}), all_rows, current_data_column);
      left_smaller = (abs(dd_table{i}(dd_table_indexes_left)) ...
                      < abs(dd_table{i}(dd_table_indexes_right)));

    end
    
    % If the left is not smaller, then the right is smaller (or equal).
    right_smaller = ~left_smaller;
    
    % In using the smaller magnitude divided difference entry, which new
    % abscissa was included?
    proposed_stencil_lcolumn = actual_stencil_lcolumn - left_smaller;
    proposed_stencil_rcolumn = actual_stencil_rcolumn + right_smaller;
    proposed_stencil_lindexes = sub2ind(size(data_invalid), all_rows, ...
                                            proposed_stencil_lcolumn);
    proposed_stencil_rindexes = sub2ind(size(data_invalid), all_rows, ...
                                             proposed_stencil_rcolumn);

    % If any invalid data values were included, we have to remove them.
    invalid_left = data_invalid(proposed_stencil_lindexes);
    invalid_right = data_invalid(proposed_stencil_rindexes);
    choose_left = (left_smaller | invalid_right) & ~invalid_left;
    choose_right = (right_smaller | invalid_left) & ~invalid_right;
    % Either left or right but not both must be valid and smaller, otherwise
    % there is a problem.
    if ~all(choose_left | choose_right)
      % There were insufficient valid entries in the stencil.
      error('Insufficient valid entries to construct a full degree interpolant.');
    end
    if any(choose_left & choose_right)
      % I don't know how, but there is a bug somewhere in the construction.
      error('Both divided difference entries cannot be used at once.');
    end
    
    % It is safe (although not necessary) to refer only to choose_left for the
    % rest of the loop because we now know that either choose_left or
    % choose_right but not both holds for all rows.

    % Note that every time we move up a level in the divided difference table,
    % the same index in data{i+1} now refers to the divided difference table
    % entry 1/2 index to the right of the same index in data{i} (because we
    % are losing table entries on the left, but Matlab indexing always starts
    % from 1).  Therefore, to get the left entries we need to subtract one
    % (which is the value Matlab conveniently assigns to true entries in
    % boolean arrays), but to get to the right entries we need do nothing.
    current_data_column = current_data_column - choose_left;

    % In using the smaller magnitude (and valid) divided difference entry, which
    % new abscissa was included?  Recalculate to take into account the
    % adjustments for invalid data values.
    actual_stencil_lcolumn = actual_stencil_lcolumn - choose_left;
    actual_stencil_rcolumn = actual_stencil_rcolumn + choose_right;

    % The new abscissa is either the leftmost or rightmost in the stencil,
    % depending on which term in the divided difference table was larger.
    current_abscissae_column(choose_left) = actual_stencil_lcolumn(choose_left);
    current_abscissae_column(choose_right) = actual_stencil_rcolumn(choose_right);
    
    % Add the next term to the interpolant.
    dd_table_indexes = sub2ind(size(dd_table{i}), all_rows, current_data_column);
    sample_y = sample_y + dd_table{i}(dd_table_indexes) .* current_basis;
    
  end % Iteration over levels in the divided difference table.

  % When we are done iterating over the levels of the divided difference table,
  % sample_y contains the desired interpolant.
  
end % main function


%---------------------------------------------------------------------------
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%---------------------------------------------------------------------------
function dd_table = dividedDifferenceTableHelper(data, levels, dx)
% dividedDifferenceTableHelper: Computes a divided difference table.
%
%   dd_table = dividedDifferenceTableHelper(data, levels, dx)
%
% Computes a divided difference table in the second dimension (across the
% columns) for a 2D array of data.  Note that no boundary conditions are
% introduced, so the width of the table shrinks by one at each level.  The
% node spacing (dx) is assumed to be constant in this implementation,
% although that could probably be fixed relatively easily.
%
% Input Parameters:
%
%   data: 2D array of data.  A divided difference table is computed separately
%   for each row.
%
%   levels: Non-negative integer.  How many levels of divided difference.
%
%   dx: Positive double.  Spacing between data entries.
%
% Output Parameters:
%
%   dd_table: Cell vector containing the divided difference table.  Each entry
%   corresponds to a level of the table.  Each entry will contain a 2D array
%   with the same number of rows as data, but the number of columns will be
%   reduced by one for each level of the table.

  %---------------------------------------------------------------------------
  %% Set up variables.

  % Create cell array to hold the divided differences.
  dd_table = cell(levels, 1);

  % In this helper routine, we always compute divided differences across the
  % second dimension.
  dim = 2;
  
  % Create cell array with array indices.
  size_data = size(data);
  indices1 = { 1 : size_data(1); 1 : size_data(2) };
  indices2 = indices1;

  %---------------------------------------------------------------------------
  %% Compute divided differences.
  last_level = data;
  for i = 1 : levels
    indices1{dim} = 2 : size(last_level, dim);
    indices2{dim} = indices1{dim} - 1;
    % This division will cause an error if dx is not a scalar.
    divisor_inv = 1 / (i * dx);
    dd_table{i} = divisor_inv * (last_level(indices1{:}) - last_level(indices2{:}));
    last_level = dd_table{i};
  end
  
end