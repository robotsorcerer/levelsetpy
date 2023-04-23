function [ data, g, data0 ] = curvatureStarDemo(accuracy,splitFlow,displayType)
% curvatureStarDemo: demonstrate motion by mean curvature on star interface.
%
%   [ data, g, data0 ] = curvatureStarDemo(accuracy, splitFlow, displayType)
%
% Recreates figure 4.2 from O&F chapter 4, showing motion by mean curvature
% of a star-shaped interface in two dimensions.  The parameters were guessed
% by trial and error.
%  
% This function was originally designed as a script file, so most of the
% options can only be modified in the file.  For example, edit the file to
% change the grid dimension, boundary conditions, flow field parameters,
% etc.
%
% Parameters:
%
%   accuracy: Controls the order of approximations.  Note that the spatial
%   approximation is always second order.
%
%                  'low'         Use odeCFL1.
%                  'medium'      Use odeCFL2 (default).
%                  'high'        Use odeCFL3.
%
%   splitFlow: Boolean.  Use the time dependent version of the flow
%   field, which freezes the right half of the domain while moving the
%   left half at double speed until the half-way time, and then freezes
%   the left half while moving the right at double speed.  Otherwise, the
%   whole domain is moved at a constant speed.  Optional.  Default = 0.
%
%   displayType: String to specify how to display results.  The specific
%   string depends on the grid dimension; look at the helper
%   visualizeLevelSet to see the options.  Optional.  Default = 'contour'.
%
% Output Parameters:
%
%   data: Implicit surface function at t_max.
%
%   g: Grid structure on which data was computed.
%
%   data0: Implicit surface function at t_0.

% Copyright 2004 Ian M. Mitchell (mitchell@cs.ubc.ca).
% This software is used, copied and distributed under the licensing 
%   agreement contained in the file LICENSE in the top directory of 
%   the distribution.
%
% Ian Mitchell, 2/13/04

%---------------------------------------------------------------------------
% You will see many executable lines that are commented out.
%   These are included to show some of the options available; modify
%   the commenting to modify the behavior.
  
%---------------------------------------------------------------------------
% Make sure we can see the kernel m-files.
%run('../addPathToKernel');

%---------------------------------------------------------------------------
% Curvature speed parameter.
bValue = 0.02;

if(nargin < 1)
  accuracy = 'medium';
end

% Use the time dependent motion?
if(nargin < 2)
  useTimeDependent = 0;
else
  useTimeDependent = splitFlow;
end

% What kind of display?
if(nargin < 3)
  displayType = 'contour';    
end

%---------------------------------------------------------------------------
% Integration parameters.
% Integration parameters.
plot_points = [ 0; 0.25; 0.5; 1.0 ];
tMax = max(plot_points);                  % End time.
t0 = 0;                                   % Start time.

% Plot at each timestep (overrides plot_points).
singleStep = 0;

% How close (relative) do we need to get to tMax to be considered finished?
small = 100 * eps;

%---------------------------------------------------------------------------
% What level set should we view?
level = 0;

% Pause after each plot?
pauseAfterPlot = 0;

% Delete previous plot before showing next?
deleteLastPlot = 0;

% Plot in separate subplots (set deleteLastPlot = 0 in this case)?
useSubplots = 1;

% If subplots are used, try to make a square set or put them all in a
% line?
subplots_in_line = 1;

%---------------------------------------------------------------------------
% Create the grid.
g.dim = 2;
g.min = -1;
g.max = +1;
g.dx = 1 / 50;
g.bdry = @addGhostExtrapolate;
g = processGrid(g);

%---------------------------------------------------------------------------
% Create initial conditions (star shaped interface centered at origin).
%   Note that in the periodic BC case, these initial conditions will not be
%   continuous across the boundary.  Regardless of boundary conditions, this
%   initial function will be far from signed distance (although it is
%   definitely an implicit surface function).  In practice, we'll just
%   ignore these little details.
points = 7;
shift = 2;
scale = 0.25;
data = zeros(size(g.xs{1}));
[ theta, r ] = cart2pol(g.xs{1}, g.xs{2});
data = r - scale * (cos(points * theta) + shift);
data0 = data;

%---------------------------------------------------------------------------
% Set up motion by mean curvature with constant coefficient.
schemeFunc = @termCurvature;
schemeData.grid = g;
schemeData.curvatureFunc = @curvatureSecond;

if(useTimeDependent)
  % Time dependent flow field, radially constant.
  %   For t <= tHalf, use high multiplier on the left.
  %   For t >  tHalf, use high multiplier on the right.
  schemeData.b = @switchValue;
  schemeData.tSwitch = 0.5 * tMax;
  schemeData.one = bValue * (1 - cos(theta));
  schemeData.two = bValue * (1 + cos(theta));
else
  % Time independent flow field is constant.
  schemeData.b = bValue;
end

%---------------------------------------------------------------------------
% Set up time approximation scheme.
integratorOptions = odeCFLset('factorCFL', 0.9, 'stats', 'on');

% Choose approximations at appropriate level of accuracy.
switch(accuracy)
 case 'low'
  integratorFunc = @odeCFL1;
 case 'medium'
  integratorFunc = @odeCFL2;
 case 'high'
  integratorFunc = @odeCFL3;
 otherwise
  error('Unknown accuracy level %s', accuracy);
end

if(singleStep)
  integratorOptions = odeCFLset(integratorOptions, 'singleStep', 'on');
end

%---------------------------------------------------------------------------
% Initialize Display
f = figure;
plot_count = 1;

% Set up subplot parameters if necessary.
if useSubplots
  if subplots_in_line
    rows = 1;
    cols = length(plot_points);
  else
    rows = ceil(sqrt(length(plot_points)));
    cols = ceil(length(plot_points) / rows);
  end
  subplot(rows, cols, plot_count);
end

if(plot_points(1) == t0)
  h = visualizeLevelSet(g, data, displayType, level, [ 't = ' num2str(t0) ]);
  plot_count = plot_count + 1;
  hold on;
  axis(g.axis);
  daspect([ 1 1 1 ]);
end

%---------------------------------------------------------------------------
% Loop until tMax (subject to a little roundoff).
tNow = t0;
startTime = cputime;
while(tMax - tNow > small * tMax)

  % Reshape data array into column vector for ode solver call.
  y0 = data(:);

  % How far to step?
  tSpan = [ tNow, plot_points(plot_count) ];
  
  % Take a timestep.
  [ t y ] = feval(integratorFunc, schemeFunc, tSpan, y0,...
                  integratorOptions, schemeData);
  tNow = t(end);

  % Get back the correctly shaped data array
  data = reshape(y, g.shape);

  if(pauseAfterPlot)
    % Wait for last plot to be digested.
    pause;
  end

  % Get correct figure.
  figure(f);

  % Delete last visualization if necessary.
  if(deleteLastPlot)
    delete(h);
  end

  % Move to next subplot if necessary.
  if(useSubplots)
    subplot(rows, cols, plot_count);
  end

  % Create new visualization.
  h = visualizeLevelSet(g, data, displayType, level, [ 't = ' num2str(tNow) ]);
  plot_count = plot_count + 1;
  
end

endTime = cputime;
fprintf('Total execution time %g seconds\n', endTime - startTime);



%---------------------------------------------------------------------------
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%---------------------------------------------------------------------------
function out = switchValue(t, data, schemeData)
% switchValue: switches between two values.
%
%  out = switchValue(t, data, schemeData)
%
% Returns a constant value:
%           one     for t <= tSwitch;
%           two     for t >  tSwitch.
%
% By setting one and two correctly, this function can implement
%   the velocityFunc prototype for termConvection;
%   the scalarGridFunc prototype for termNormal, termCurvature and others;
%   and possibly some other prototypes...
%
% Parameters:
%   t            Current time.
%   data         Level set function.
%   schemeData   Structure (see below).
%
%   out          Either schemeData.one or schemeData.two.
%
% schemeData is a structure containing data specific to this type of 
%   term approximation.  For this function it contains the field(s)
%
%   .one         The value to return for t <= tSwitch.
%   .two         The value to return for t >  tSwitch.
%   .tSwitch     The time at which the switch between flow fields occurs.
%
% schemeData may contain other fields.

  checkStructureFields(schemeData, 'one', 'two', 'tSwitch');
  
  if(t <= schemeData.tSwitch)
    out = schemeData.one;
  else
    out = schemeData.two;
  end
