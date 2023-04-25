function [ mttr, attr, data, gridOut, data0 ] = ...
                                          doubleIntegratorTTR(accuracy, gridIn)
% doubleIntegratorTTR: demonstrate the double integrator time to reach.
%
%   [ mttr, attr, data, gridOut, data0 ] = doubleIntegratorTTR(accuracy,gridIn)
%  
% In this example we calculate the minimum time to reach a small ball
%   around the origin under standard double integrator dynamics with
%   bounded input magnitude.
%
% The dynamics are
%
%        \dot x    = y
%	 \dot y    = b
%
%   where input |b| \leq 1 is trying to hit the target.
%
% The goal is slightly more ambitious than the standard reach set, since
%   we would also like to record the minimum time to reach function by
%   tracking the time at which the reach set arrives at each node.
%
% This function was originally designed as a script file, so most of the
%   options can only be modified in the file.
%
% Reinitialization is available, but only seems to degrade solution quality.
%
% For example, edit the file to change the grid dimension, boundary conditions,
%   aircraft parameters, etc.
%
% Parameters:
%
%   accuracy     Controls the order of approximations.
%                  'low'         Use odeCFL1 and upwindFirstFirst.
%                  'medium'      Use odeCFL2 and upwindFirstENO2 (default).
%                  'high'        Use odeCFL3 and upwindFirstENO3.
%                  'veryHigh'    Use odeCFL3 and upwindFirstWENO5.
%   gridIn       Computational grid.
%                  Default is dim = 2; min = -1; max = +1; N = 101;
%                  and bdry = @addGhostExtrapolate.
%
%   mttr         Minimum time to reach function at t_max.
%   attr         Analytic time to reach function.
%                  Determined by calling analyticDoubleIntegratorTTR.
%   data         Implicit surface function at t_max for the reach set.
%   gridOut      Grid structure on which data was computed.
%   data0        Implicit surface function at t_0.

% Copyright 2004 Ian M. Mitchell (mitchell@cs.ubc.ca).
% This software is used, copied and distributed under the licensing 
%   agreement contained in the file LICENSE in the top directory of 
%   the distribution.
%
% Ian Mitchell, 9/20/04

%---------------------------------------------------------------------------
% You will see many executable lines that are commented out.
%   These are included to show some of the options available; modify
%   the commenting to modify the behavior.

%---------------------------------------------------------------------------
% Make sure we can see the kernel m-files.
run('setup.m');

%---------------------------------------------------------------------------
% Integration parameters.
tMax = 2.0;                  % End time.
plotSteps = 9;               % How many intermediate plots to produce?
t0 = 0;                      % Start time.
singleStep = 0;              % Plot at each timestep (overrides tPlot).

% Period at which intermediate plots should be produced.
tPlot = (tMax - t0) / (plotSteps - 1);

% How close (relative) do we need to get to tMax to be considered finished?
small = 100 * eps;

% What kind of dissipation?
dissType = 'global';

%---------------------------------------------------------------------------
% Problem Parameters.

% Bounds on the input magnitude.
inputBound = 1;

% Radius of the target set.
%   A value of zero will be translated into a value of (smallTarget * g.dx).
targetRadius = 0.2;

% What is considered a small target?
smallTarget = 1.0;

% Should we reinitialize after every step?
reinitialize = 0;

% What type of initial conditions to use?
whichIC = 'analytic';
whichIC = 'square';
whichIC = 'circle';

% File where signed distance version of IC is stored.
fileIC = 'doubleIntSignedDist';

%---------------------------------------------------------------------------
% What level set should we view?
level = 0;

% Visualize the reachable set.
displayType = 'contour';

% Pause after each plot?
pauseAfterPlot = 0;

% Delete previous plot before showing next?
deleteLastPlot = 1;

% Plot in separate subplots (set deleteLastPlot = 0 in this case)?
useSubplots = 0;

% Show analytic solution for comparison?
showAnalytic = 1;

%---------------------------------------------------------------------------
% Create the grid (if necessary).

if(nargin < 2)
  g.dim = 2;
  g.min = -1;
  g.max = +1;
  g.bdry = @addGhostExtrapolate;
  g.N = 101;
else
  g = gridIn;
end

g = processGrid(g);

if(nargout > 3)
  gridOut = g;
end

% Check the target radius
if(targetRadius == 0)
  % Precisely zero target radius will not work.  
  %   Make it small relative to the size of the grid.
  targetRadius = smallTarget * max(g.dx);
end

%---------------------------------------------------------------------------
% Evaluate analytic solution for target of appropriate radius.
attr = analyticDoubleIntegratorTTR(g) - targetRadius;

% Create initial conditions.
switch(whichIC)
 case 'circle'
  data = shapeSphere(g, zeros(g.dim, 1), targetRadius);

 case 'square'
  data = shapeRectangleByCenter(g, zeros(g.dim, 1), ...
                                2 * targetRadius * ones(g.dim,1));

 case 'analytic'
  data = attr;

 otherwise
  error('Unknown initial condition whichIC = %s', whichIC);
end

if(nargout > 4)
  data0 = data;
end

% Although we needed negative values in attr to properly define the
%   implicit surface for the 'analytic' IC case, real time to reach
%   functions do not have negative values.
attr = max(0, attr);

%---------------------------------------------------------------------------
% Set up spatial approximation scheme.
schemeFunc = @termLaxFriedrichs;
schemeData.hamFunc = @doubleIntegratorHamFunc;
schemeData.partialFunc = @doubleIntegratorPartialFunc;
schemeData.grid = g;

% The Hamiltonian and partial functions need problem parameters.
schemeData.inputBound = inputBound;

%---------------------------------------------------------------------------
% Choose degree of dissipation.

switch(dissType)
 case 'global'
  schemeData.dissFunc = @artificialDissipationGLF;
 case 'local'
  schemeData.dissFunc = @artificialDissipationLLF;
 case 'locallocal'
  schemeData.dissFunc = @artificialDissipationLLLF;
 otherwise
  error('Unknown dissipation function %s', dissFunc);
end

%---------------------------------------------------------------------------
if(nargin < 1)
  accuracy = 'medium';
end

% Set up time approximation scheme.
integratorOptions = odeCFLset('factorCFL', 0.75, 'stats', 'on');

% Choose approximations at appropriate level of accuracy.
switch(accuracy)
 case 'low'
  schemeData.derivFunc = @upwindFirstFirst;
  integratorFunc = @odeCFL1;
 case 'medium'
  schemeData.derivFunc = @upwindFirstENO2;
  integratorFunc = @odeCFL2;
 case 'high'
  schemeData.derivFunc = @upwindFirstENO3;
  integratorFunc = @odeCFL3;
 case 'veryHigh'
  schemeData.derivFunc = @upwindFirstWENO5;
  integratorFunc = @odeCFL3;
 otherwise
  error('Unknown accuracy level %s', accuracy);
end

if(singleStep)
  integratorOptions = odeCFLset(integratorOptions, 'singleStep', 'on');
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
% Set up minimum time to reach recording using postTimestepFunc.
integratorOptions = odeCFLset(integratorOptions, ...
                              'postTimestep', @postTimestepTTR);

% Initialize the minimum time to reach function by calling
%   the postTimestepFunc once with initial data.
y = data(:);
[ y, schemeData ] = feval(@postTimestepTTR, t0, y, schemeData);
data = reshape(y, g.shape);

%---------------------------------------------------------------------------
if(reinitialize)

  % We need some more parameters to drive the reinitialization.
  %   But most can just be the defaults.
  schemeData.grid = g;
  schemeData.reinitAccuracy = accuracy;
  schemeData.reinitSteps = 1;

  % Register the entire cell vector of postTimestep routines.
  integratorOptions = odeCFLset(integratorOptions, 'postTimestep', ...
                                { @postTimestepReinit; @postTimestepTTR });
end

%---------------------------------------------------------------------------
% Initialize Display
%f = figure;
f = figure(1); clf;

% Set up subplot parameters if necessary.
if(useSubplots)
  rows = ceil(sqrt(plotSteps));
  cols = ceil(plotSteps / rows);
  plotNum = 1;
  subplot(rows, cols, plotNum);
end

h = visualizeLevelSet(g, data, displayType, level, [ 't = ' num2str(t0) ]);

hold on;

if(showAnalytic)
  ha = [];
end

%---------------------------------------------------------------------------
% Loop until tMax (subject to a little roundoff).
tNow = t0;
startTime = cputime;

avg_time = [];
while(tMax - tNow > small * tMax)
  internal_time_start = cputime;

  % Reshape data array into column vector for ode solver call.
  y0 = data(:);

  % How far to step?
  tSpan = [ tNow, min(tMax, tNow + tPlot) ];
  
  % Take a timestep.
  [ t, y, schemeData ] = feval(integratorFunc, schemeFunc, tSpan, y0,...
                               integratorOptions, schemeData);
  tNow = t(end);

  fprintf('{tNow/T}: {%.2f/%.2f} | Targ bnds: {%.4f/%.4f} | norm (y): %.4f\n', tNow, tMax, min(y), max(y), norm(y));
  % Get back the correctly shaped data array
  data = reshape(y, g.shape);

  if(pauseAfterPlot)
    % Wait for last plot to be digested.
    pause;
  end

  % Get correct figure, and remember its current view.
  figure(f);

  % Delete last visualization if necessary.
  if(deleteLastPlot)
    delete(h);
  end
   
  % Move to next subplot if necessary.
  if(useSubplots)
    plotNum = plotNum + 1;
    subplot(rows, cols, plotNum);
  end

  if(showAnalytic)
    delete(ha);
    ha = visualizeLevelSet(g, attr, displayType, tNow);
    set(ha, 'LineStyle', '--');
  end

  % Create new visualization.
  h = visualizeLevelSet(g, data, displayType, level, [ 't = ' num2str(tNow) ]);

  internal_time_end = cputime;
  avg_time = [avg_time, internal_time_end-internal_time_start];
end

endTime = cputime;
local_time = sum(avg_time)/(length(avg_time));
global_time = endTime - startTime;

fprintf('Avg local time: %g.\t', local_time);
fprintf('Total execution time %g seconds\n', global_time);

% Extract the minimum time to reach function from the schemeData structure.
%   Reshape it into an array, and replace the +inf entries with NaN to
%   make the visualization more pleasant.
% mttr = reshape(schemeData.ttr, g.shape);
% unreached = find(isinf(mttr));
% mttr(unreached) = NaN;



%---------------------------------------------------------------------------
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%---------------------------------------------------------------------------
function hamValue = doubleIntegratorHamFunc(t, data, deriv, schemeData)
% doubleIntegratorHamFunc: analytic Hamiltonian for the double integrator.
%
% hamValue = doubleIntegratorHamFunc(t, data, deriv, schemeData)
%
% This function implements the hamFunc prototype for the double integrator:
%
%   hamValue = -(x_2 * deriv{1} + schemeData.inputBound * abs(deriv{2}))
%
% Parameters:
%   t            Time at beginning of timestep (ignored).
%   data         Data array.
%   deriv	 Cell vector of the costate (\grad \phi).
%   schemeData	 A structure (see below).
%
%   hamValue	 The analytic hamiltonian.
%
% schemeData is a structure containing data specific to this Hamiltonian
%   For this function it contains the field(s):
%
%   .grid	 Grid structure.
%   .inputBound  Bound on the magnitude of the input.
%
% Ian Mitchell 9/20/04

checkStructureFields(schemeData, 'grid', 'inputBound');

hamValue = -(schemeData.grid.xs{2} .* deriv{1} - ...
             schemeData.inputBound .* abs(deriv{2}));



%---------------------------------------------------------------------------
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%---------------------------------------------------------------------------
function alpha = ...
      doubleIntegratorPartialFunc(t, data, derivMin, derivMax, schemeData, dim)
% doubleIntegratorPartialFunc: partial fcn for the double integrator.
%
%   alpha = doubleIntegratorPartialFunc(t, data, derivMin, ...
%                                       derivMax, schemeData, dim)
%
% This function implements the partialFunc prototype for the double integrator.
%
% It calculates the extrema of the absolute value of the partials of the 
%   analytic Hamiltonian with respect to the costate (gradient).
%
% Referring to the Hamiltonian from the previous function, those partials are:
%
%   in dimension 1: abs(x_2)
%   in dimension 2: schemeData.inputBound
%
% Parameters:
%   t            Time at beginning of timestep (ignored).
%   data         Data array.
%   derivMin	 Cell vector of minimum values of the costate (\grad \phi).
%   derivMax	 Cell vector of maximum values of the costate (\grad \phi).
%   schemeData	 A structure (see below).
%   dim          Dimension in which the partial derivatives is taken.
%
%   alpha	 Maximum absolute value of the partial of the Hamiltonian
%		   with respect to the costate in dimension dim for the 
%                  specified range of costate values (O&F equation 5.12).
%		   Note that alpha can (and should) be evaluated separately
%		   at each node of the grid.
%
% schemeData is a structure containing data specific to this Hamiltonian
%   For this function it contains the field(s):
%
%   .grid	 Grid structure.
%   .inputBound  Bound on the magnitude of the input.
%
% Ian Mitchell 9/20/04

checkStructureFields(schemeData, 'grid', 'inputBound');

switch dim
  case 1
    alpha = abs(schemeData.grid.xs{2});

  case 2
    alpha = abs(schemeData.inputBound);

  otherwise
    error([ 'Partials for the double integrator '...
            'only exist in dimensions 1-2' ]);
end
