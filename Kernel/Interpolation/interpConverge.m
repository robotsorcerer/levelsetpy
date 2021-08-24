% Script file to demonstrate the convergence rate of the non-oscillatory
% interpolation schemes.
%
% Uses interpTest1 for a sequence of approximation schemes and
% grid sizes.
%
% Also demonstrates Matlab's bizarre but useful method of accessing
% structure fields by string variables.

% Copyright 2011 Ian M. Mitchell (mitchell@cs.ubc.ca).
%
% This software is used, copied and distributed under the licensing 
% agreement contained in the file LICENSE in the top directory of 
% the distribution.
%
% Ian Mitchell, 6/29/2011
% $Date: 2011-12-12 16:24:22 -0800 (Mon, 12 Dec 2011) $
% $Id: interpConverge.m 71 2011-12-13 00:24:22Z mitchell $

% Need to make sure that the Kernel is on the path.
% run('../addPathToKernel');

  methods = { 'linear', 'eno2', 'cubic', 'eno3', 'eno4' };
  % Be careful with grid dx that you don't overwhelm the memory
  grid_sizes = [ 20; 40; 80; 160; 320; 640; 1280 ];
  %grid_sizes = [ 20; 40; 80; 160 ];
  dim = 1;
  which_dims = +1;

  % Choose plotting styles for the various methods.
  method_styles = { 'r--v', 'm-*', 'b--x', 'c-o', 'k-+' };
  
  % File id for a place to report the statistics (the screen).
  screen = 1;

  % We will use a random number stream that is reset after each
  % interpolation test so that the same samples are drawn for each test.
  % This has a side-effect that we draw the same samples every time the
  % script is run.  If you want to avoid that, then create a new random
  % stream with a seed based on, for example, the clock.  See Matlab's
  % RandStream for details.
  rand_stream = RandStream.getDefaultStream;
  
  % Create some figures to display the results.
  error_types = { 'maximum', 'average', 'rms', 'jumps' };
  for i = 1 : length(error_types)
    error_type = error_types{i};
    f.(error_type) = figure;
    title([ 'Error analysis for ' error_type ' error.' ]);
    hold on;
  end

  h = zeros(length(error_types), length(methods));
  times = zeros(length(methods), 1);
  fprintf(screen, 'For grid size %d in %d dimensions, execution time (sec)\n',...
          grid_sizes(end), dim);
        
  for m = 1 : length(methods);
    method = methods{m};

    % Collect the error statistics.  Only keep track of the running time
    % for the final (assumed largest) grid.
    for n = 1 : length(grid_sizes);
      grid_size = grid_sizes(n);
      % Reset the random number stream to ensure that we pick the same
      % samples on each run.  Note that different samples will be drawn for
      % different grid sizes, because a different number of samples is
      % drawn for different grid sizes.
      rand_stream.reset;
      % In order to preallocate this array of structures, we would have to
      % use exactly the same structure names as are returned by
      % interpTest1, which seems prone to error.  Instead, we'll just eat
      % the overhead of reallocation.
      [ error_stats(n), times(m) ] = interpTest1(method, dim, which_dims, 1.0 / grid_size, rand_stream); %#ok<SAGROW>
    end

    % Plot the error statistics.
    for i = 1 : length(error_types)
      error_type = error_types{i};
      figure(f.(error_type));
      h(i, m) = plot(grid_sizes, [ error_stats(:).(error_type) ], method_styles{m});
    end

    fprintf(screen, '\t%s\t%f\n', methods{m}, times(m));
  end

  % Make the plots a little prettier.
  for i = 1 : length(error_types)
    error_type = error_types{i};
    figure(f.(error_type));

    % Clean up the axes.
    set(gca, 'XScale', 'log', 'YScale', 'log', 'XTick', grid_sizes, ...
             'XGrid', 'on', 'XMinorGrid', 'off', ...
             'YGrid', 'on', 'YMinorGrid', 'off');
    xlabel('Grid Size');  ylabel('Error');

    % Set the axes so that they include the data (plus some space vertically).
    curr_axis = axis;
    new_axis = [ min(grid_sizes), max(grid_sizes), ...
                10^(floor(log10(curr_axis(3)))), 10^(ceil(log10(curr_axis(4)))) ];
    axis(new_axis);

    % Show slopes with various orders of convergence.
    slope_point = [ grid_sizes(end-2); 100 * new_axis(3) ];
    slope_handles = addSlopes(slope_point, grid_sizes(end-1) - grid_sizes(end-2), 'k-', ...
                             [ 0, -1, -2 -3 -4 ], ...
                             { '', '1', '2', '3', '4' });

    % Label the schemes.
    legend(h(i, :), methods, 3);

    % Make the lines and markers a bit more visible.
    set(h(i, :), 'LineWidth', 2.0, 'MarkerSize', 9.0);
  end
