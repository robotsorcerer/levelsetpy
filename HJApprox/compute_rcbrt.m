function [ data, g, data0 ] = compute_rcbrt(model)
  % compute_rcbrt: Compute an RCBRT for a given model.
  %
  %   [ data, g, data0 ] = compute_rcbrt(rcbrt)
  %  
  % This function was originally designed as a script file, so most of the
  % options can only be modified in the file.
  %
  % Input Parameters:
  %
  %   model: Default = BasicModel().  A model object for an RCBRT calculation.
  %
  % Output Parameters:
  %
  %   data: Implicit surface function at t_max.
  %
  %   g: Grid structure on which data was computed.
  %
  %   data0: Implicit surface function at t_0.

  % Created by Ian Mitchell, 2020-05-18

  %---------------------------------------------------------------------------
  % You will see many executable lines that are commented out.
  %   These are included to show some of the options available; modify
  %   the commenting to modify the behavior.

  if(nargin < 1)
    model = BasicModel();
  end

  %---------------------------------------------------------------------------
  % Integration parameters.
  plot_steps = 11;              % How many intermediate plots to produce?
  single_step = 0;              % Plot at each timestep (overrides t_plot).

  % How close (relative) do we need to get to tMax to be considered finished?
  small = 100 * eps;

  %---------------------------------------------------------------------------
  % What level set should we view?
  level = 0;

  % Pause after each plot?
  pause_after_plot = false;

  % Delete previous plot before showing next?
  delete_last_plot = true;

  % Plot in separate subplots (set delete_last_plot = false in this case)?
  use_subplots = false;
  
  %---------------------------------------------------------------------------
  % Get the grid, initial conditions and time interval from the model.
  g = model.getGrid();
  data0 = model.getTarget();
  t_range = model.getTimeInterval();
  
  data = data0;

  %---------------------------------------------------------------------------
  % Type of visualization depends on the dimension of the problem.
  switch(g.dim)
    case 2
      display_type = 'contour';
    case 3
      display_type = 'surface';
    otherwise
      error('Do not know how to visualize intermediate results in dimension %d', g.dim);
  end
  
  %---------------------------------------------------------------------------
  % Set up level set approximation scheme.
  schemeFunc = @termLaxFriedrichs;
  schemeData.hamFunc = @(t, data, deriv, schemeData) ...
    model.hamFunc(t, data, deriv, schemeData);
  schemeData.partialFunc = @(t, data, derivMin, derivMax, schemeData, dim) ...
    model.partialFunc(t, data, derivMin, derivMax, schemeData, dim);
  schemeData.grid = g;
  schemeData.dissFunc = @artificialDissipationGLF;
  % Change the integer to 3 for slightly higher accuracy.
  schemeData.derivFunc = @upwindFirstENO2;
  integratorFunc = @odeCFL2;
  % Set up time approximation scheme.
  integratorOptions = odeCFLset('factorCFL', 0.95, 'stats', 'on');

  if(single_step)
    integratorOptions = odeCFLset(integratorOptions, 'singleStep', 'on'); %#ok<UNRCH>
  end

  %---------------------------------------------------------------------------
  % Restrict the Hamiltonian so that reachable set only grows.
  %   The Lax-Friedrichs approximation scheme MUST already be completely set up.
  innerFunc = schemeFunc;
  innerData = schemeData;
  clear schemeFunc schemeData;

  % Wrap the true Hamiltonian inside the term approximation restriction routine.
  schemeFunc = @termRestrictUpdate;
  schemeData.innerFunc = innerFunc;
  schemeData.innerData = innerData;
  schemeData.positive = 0;

  %---------------------------------------------------------------------------
  % Initialize Display
  f = figure;

  % Period at which intermediate plots should be produced.
  t_plot = (t_range(2) - t_range(1)) / (plot_steps - 1);
  
  % Set up subplot parameters if necessary.
  if(use_subplots)
    rows = ceil(sqrt(plot_steps));  %#ok<UNRCH>
    cols = ceil(plot_steps / rows);
    plot_num = 1;
    subplot(rows, cols, plot_num);
  end

  h = visualizeLevelSet(g, data, display_type, level, [ 't = ' num2str(t_range(1)) ]); 

  camlight right;  camlight left;
  hold on;
  axis(g.axis);
  daspect([ 1 1 2 ])
  drawnow;

  %---------------------------------------------------------------------------
  % Loop through t_range (subject to a little roundoff).
  t_now = t_range(1);
  start_time = cputime;
  while(t_range(2) - t_now > small * t_range(2))

    % Reshape data array into column vector for ode solver call.
    y0 = data(:);

    % How far to step?
    t_span = [ t_now, min(t_range(2), t_now + t_plot) ];

    % Take a timestep.
    [ t, y ] = integratorFunc(schemeFunc, t_span, y0, integratorOptions, schemeData);
    t_now = t(end);

    % Get back the correctly shaped data array
    data = reshape(y, g.shape);

    if(pause_after_plot)
      % Wait for last plot to be digested.
      pause; %#ok<UNRCH>
    end

    % Get correct figure, and remember its current view.
    figure(f);
    [ view_az, view_el ] = view;

    % Delete last visualization if necessary.
    if(delete_last_plot)
      delete(h); 
    end

    % Move to next subplot if necessary.
    if(use_subplots)
      plot_num = plot_num + 1; %#ok<UNRCH>
      subplot(rows, cols, plot_num);
    end

    % Create new visualization.
    h = visualizeLevelSet(g, data, display_type, level, [ 't = ' num2str(t_now) ]);

    % Restore view.
    view(view_az, view_el);

  end

  end_time = cputime;
  fprintf('Total execution time %g seconds\n', end_time - start_time);

end % rcbrt().