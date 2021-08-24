function [ errors, time ] = interpTest1(method, dim, which_dims, dx, rand_stream)
% interpTest1: test non-oscillatory interpolation schemes
%
%   [ errors, time ] = interpTest1(method, dim, which_dims, dx, rand_stream)
%
% Function to test the various non-oscillatory interpolation schemes.
%
% Uses a concatenation of shifted sin functions for data so as to introduce
% three derivative discontinuities (including one at the boundary in the
% periodic BC case):
%
%          / sin(2*pi*z + pi/4),  for 0.0  \leq z < 0.25
%   f(z) = | sin(2*pi*z - pi/4),  for 0.25 \leq z < 0.5
%          \ sin(2*pi*z) + 1,     for 0.5  \leq z < 1.0
%
% An interpolation location is chosen randomly in each cell of the grid.
%
% If no output parameters are requested, the function produces text output
% explaining the error.  In this case if dim \in { 1, 2 } as well, a figure
% is also generated showing the function and interpolants.  For plotting
% purposes, the function is sampled at five times the grid resolution.
%
% Input Parameters:
%
%   method: String.  Which interpolation scheme to use.  Options are the
%   same as for internNO (which is the function used to evaluate the
%   interpolants).
%
%   dim: Positive integer.  How many dimensions should the computational
%   grid be?  Optional.  Default = 1.
%
%   which_dims: Integer or vector.  What combination of dimensions should
%   be used to supply the variable z in the function definition.  If
%   which_dims is a positive integer, then z = x_{which_dims}.  If
%   which_dims is a negative ineteger, then z = -x_{which_dims}.  If
%   which_dims is a vector of positive integers, then z is the sum of the
%   corresponding dimensions; negative integers indicate that the
%   corresponding dimension should be multiplied by -1.  Optional.  Default
%   = +1.
%
%   dx: Double.  Grid spacing.  The derivative discontinuities occur at
%   0, 0.25 and 0.5, so it is best to choose a spacing which will put
%   samples at these points.  Optional.  Default = 0.025.
%
%   rand_stream: Random number stream to use for generating the random
%   points at which the interpolant is sampled.  See Matlab's RandStream
%   for details.  Optional.  Defaults to the default random number stream
%   (ie: RandStream.getDefaultStream).  If you want to reset the stream so that
%   you get the same sequence again, use the reset method for RandStream
%   (eg: reset(RandStream.getDefaultStream)).
%
% Output Parameters:
%
%   errors: structure containing information about the error.  The following
%   fields are supplied:
%
%     .maximum: maximum error (inf norm).
%     .average: average error (one norm).
%     .rms: root mean square error (two norm).
%     .jump: average error at the three jumps.
%
%   time: Double.  Execution time of the interpolation routine in seconds
%   (not including any time to initialize the input data or analyze the
%   results).

% Copyright 2011 Ian M. Mitchell (mitchell@cs.ubc.ca).
% This software is used, copied and distributed under the licensing 
%   agreement contained in the file LICENSE in the top directory of 
%   the distribution.
%
% Ian Mitchell, 6/21/2011
% $Date: 2011-12-12 16:24:22 -0800 (Mon, 12 Dec 2011) $
% $Id: interpTest1.m 71 2011-12-13 00:24:22Z mitchell $

%---------------------------------------------------------------------------
%% Optional parameters.

  if(nargin < 2)
    dim = 1;
  end
  if(nargin < 3)
    which_dims = +1;
  end
  if(nargin < 4)
    dx = 0.025;
  end
  if(nargin < 5)
    rand_stream = RandStream.getDefaultStream;
  end

  %---------------------------------------------------------------------------
  %% Other parameters.

  % Periodic BC?
  periodic = 1;

  % File id for a place to report the statistics (the screen).
  screen = 1;

  %---------------------------------------------------------------------------
  %% Build a grid.
  g.dim = dim;
  g.min = zeros(g.dim, 1);
  g.dx = dx;
  if(periodic)
    g.max = (1 - g.dx) * ones(g.dim, 1);
    g.bdry = @addGhostPeriodic;
  else
    g.max = ones(g.dim, 1);
    g.bdry = @addGhostExtrapolate;
  end
  g = processGrid(g);

  %---------------------------------------------------------------------------
  %% Determine random interpolation sample points.
  
  % Construct a cell matrix containing the interpolation points by starting
  % from the node in the lower left corner of each grid cell, and adding a
  % random offset within that cell in each dimension.
  indexes = cell(g.dim);
  for d = 1 : g.dim
    indexes{d} = 1 : g.N(d) - 1;
  end
  interp_x = cell(g.dim, 1);
  % We need a special case because Matlab doesn't handle 1D arrays.
  if(g.dim > 1)
    for d = 1 : g.dim
      interp_x{d} = g.xs{d}(indexes{:}) + g.dx(d) * rand_stream.rand(g.shape - 1);
    end
  else
    interp_x{1} = g.xs{1}(indexes{:}) + g.dx(1) * rand_stream.rand(g.N - 1, 1);
  end
  
  % In the future, it might be nice to sample multiple points in each cell.
  
  %---------------------------------------------------------------------------
  %% Evaluate the function.
  
  % What is the input variable to our function?
  data_z = buildInputVariable(g.xs, which_dims);
  interp_z = buildInputVariable(interp_x, which_dims);

  % Evaluate the function at the grid nodes and at the sample points.
  data_f = testFunction(data_z);
  [ interp_f, jumps_f ] = testFunction(interp_z);
  
  %---------------------------------------------------------------------------
  %% Calculate the interpolants.

  % No need to supply an extrapolation value: By construction there should
  % be no extrapolation.
  start_time = cputime;
  interp_result = interpnNO(g, data_f, interp_x, method);
  end_time = cputime;
  time = end_time - start_time;

  %---------------------------------------------------------------------------
  %% Compute the error.
  error_vector = interp_result - interp_f;
  n = numel(error_vector);
  jump_error = error_vector(jumps_f);
  jump_error = jump_error(:);
  
  errors.maximum = norm(error_vector(:), inf);
  errors.average = norm(error_vector(:), 1) / n;
  errors.rms = norm(error_vector(:), 2) / sqrt(n);
  errors.jumps = norm(jump_error, 1) / sum(jumps_f);
  
  %---------------------------------------------------------------------------
  %% Generate visual output if there are no output arguments.  
  if(nargout == 0)
    % Textual output is always possible.
    fprintf(screen, 'Computational time %f\n', time);
    fprintf(screen, 'Error: max %g, mean %g, mean at jumps %g\n', ...
            errors.maximum, errors.average, errors.jumps);

    % One dimensional grids are easy to plot.
    if(g.dim == 1)
      figure;
      subplot(2,1,1);
      plot(data_z, data_f, 'b.');
      hold on;
      plot(interp_z, interp_result, 'kx');
      plot(interp_z(jumps_f), interp_result(jumps_f), 'r*');

      % Supersample the function.
      g2.dim = g.dim;
      g2.max = g.max;
      g2.min = g.min;
      g2.bdry = g.bdry;
      g2.dx = 0.2 * g.dx;
      g2 = processGrid(g2);
      data2_z = buildInputVariable(g2.xs, which_dims);
      data2_f = testFunction(data2_z);
      plot(data2_z, data2_f, 'b-');

      grid on;
      axis([ g.min, g.max, -0.05, +1.05 ]);

      % Error plot.
      subplot(2,1,2)
      plot(interp_z, error_vector, 'kx');
      hold on;
      plot(interp_z(jumps_f), error_vector(jumps_f), 'r*');
      grid on;
    end
    
    % Two dimensional grids are a little trickier.
    if(g.dim == 2)
      figure;
      surf(g.xs{:}, data_f);
      hold on;
      plot3(interp_x{:}, interp_result, 'kx');
      jumps_x = cell(g.dim, 1);
      for d = 1 : g.dim
        jumps_x{d} = interp_x{d}(jumps_f);
      end
      plot3(jumps_x{:}, interp_result(jumps_f), 'r*');
      
      grid on;
      axis([ g.axis, -0.05, +1.05 ]);
    end
    
  end % if(nargout == 0).
  
end % interpTest1().

%---------------------------------------------------------------------------
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%---------------------------------------------------------------------------
function zs = buildInputVariable(xs, which_dims)
% Compute the variable which will be passed to the test function.
%
% Input parameters:
%
%   xs: Cell vector.  Grid on which to build the input variable.  For
%   example, it could be the grid.xs{} cell vector.
%
%   which_dims: Integer or vector of integers.  Same rules as for
%   interpTest1.
%
% Output Parameters:
%
%   zs: Array the same size as the elements of xs.  The input variable's
%   value at each point in the array defined by xs.

  zs = zeros(size(xs{1}));
  array_dim = ndims(zs);
  for d = 1 : numel(which_dims)
    this_dim = which_dims(d);
    if((this_dim > 0) && (this_dim <= array_dim))
      zs = zs + xs{this_dim};
    elseif((this_dim < 0) && (this_dim >= -array_dim))
      zs = zs - xs{-this_dim};
    else
      error('Invalid which_dims dimension: %d', this_dim);
    end
  end


end % buildInputVariable().


%---------------------------------------------------------------------------
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%---------------------------------------------------------------------------
function [ f, jumps ] = testFunction(zs)
% The function which is being interpolated.
%
% Input parameters:
%
%   zs: Double array.  Points at which to evaluate the test function.  Note
%   that this routine will not work if any of the dimensions of zs are of
%   size 1 (unless the problem is for a 1D array, in which case the second
%   dimension can be 1 because of stupid Matlab's no-1D-array rule).
%
% Output parameters:
%
%   f: Double array, same size as zs.  The function at each of these points.
%
%   jumps: Boolean array, same size as zs.  Indicates which samples are
%   nearest to the derivative discontinuities in the test function.

  % A reasonably small number.
  small = sqrt(eps);

  % Figure out where each of the three subfunctions are applicable.
  num_jumps = 3;
  indicator = cell(num_jumps, 1);
  indicator{1} = (zs - 0.25 < -small);
  indicator{2} = ((zs - 0.25 > -small) & (zs - 0.5 < -small));
  indicator{3} = (zs - 0.5 > -small);

  % Evaluate the test function.
  zs2pi = 2 * pi * zs;
  f = (indicator{1} .* sin(zs2pi + 0.5 * pi) ...
       + indicator{2} .* sin(zs2pi - 0.5 * pi) ...
       + indicator{3} .* (sin(zs2pi) + 1));

  % Computing the jump array is rather involved, so we will only do it if
  % necessary.
  if(nargout > 1)
    
    % We need some information about the shape of the sample array.
    size_zs = size(zs);
    dim = ndims(zs);
    if((dim == 2) && (size_zs(2) == 1))
      % This is really a 1D array, not a 2D array.
      dim = 1;
    end
    
    % We need indexes into the sample array.
    indexes_L = cell(dim, 1);
    for d = 1 : dim
      indexes_L{d} = 1 : size_zs(d);
    end
    indexes_R = indexes_L;

    % A sample is considered to be nearest to a jump if its neighbour in
    % any single dimension has a different value of any indicator function.
    % Note that this test does not include diagonal neighbours (eg: there
    % are only 2*dim neighbours for each sample).
    jumps = false(size(zs));
    for d = 1 : dim
      % Test against neighbours in dimension d.
      indexes_R{d} = [ 2 : size_zs(d), 1 ];
      for j = 1 : num_jumps
        disagree = (indicator{j}(indexes_L{:}) ~= indicator{j}(indexes_R{:}));
        jumps(indexes_L{:}) = jumps(indexes_L{:}) | disagree;
        jumps(indexes_R{:}) = jumps(indexes_R{:}) | disagree;
      end
      % Reset indexes for dimension d before moving on to the next
      % dimension.
      indexes_R{d} = 1 : size_zs(d);
    end
  end % if(nargout > 1).
  
end % testFunction().
