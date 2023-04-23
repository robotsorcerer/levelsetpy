function [ data, g, data0 ] = reinitTest(initialType, accuracy, displayType)
% reinitTest: test signedDistanceIterative.
%
%   [ data, g, data0 ] = reinitTest(initialType, accuracy, displayType)
%
% Demonstrates how signedDistanceIterative can be used to turn one of
%   several different dynamic surface functions into signed distance
%   function.  While it works in any dimension, it is hard to visualize the
%   difference in dimensions higher than two.
%  
% This function was originally designed as a script file, so most of the
%   options can only be modified in the file.
%
% For example, edit the file to change the grid dimension, boundary conditions,
%   flow field parameters, etc.
%
% Parameters:
%
%   initialType  String to specify which initial dynamic implicit surface.
%                  'circle'      An off center circle/sphere (default).
%                  'star'        A star-shaped interface (not 1D).
%   accuracy     Controls the order of approximations.
%                  'low'         Use odeCFL1 and upwindFirstFirst.
%                  'medium'      Use odeCFL2 and upwindFirstENO2 (default).
%                  'high'        Use odeCFL3 and upwindFirstENO3.
%                  'veryHigh'    Use odeCFL3 and upwindFirstWENO5.
%   displayType  String to specify how to display results.
%                  The specific string depends on the grid dimension;
%                  look at the subfunction visualize to see the options
%                  (optional, default depends on grid dimension).
%
%   data         Implicit surface function at t_max.
%   g            Grid structure on which data was computed.
%   data0        Implicit surface function at t_0.

% Copyright 2004 Ian M. Mitchell (mitchell@cs.ubc.ca).
% This software is used, copied and distributed under the licensing 
%   agreement contained in the file LICENSE in the top directory of 
%   the distribution.
%
% Ian Mitchell, 2/14/04

%---------------------------------------------------------------------------
% You will see many executable lines that are commented out.
%   These are included to show some of the options available; modify
%   the commenting to modify the behavior.
  
%---------------------------------------------------------------------------
% Make sure we can see the kernel m-files.
run('Examples/addPathToKernel');

%---------------------------------------------------------------------------
% What level set should we view?
level = 0;

% Across how many grid cells should we reinitialize?
%   (Choose inf to reinitialize to completion).
reinitGridCells = 20;

% What is the convergence criterion?
errorMax = 1e-3;

%---------------------------------------------------------------------------
% Default parameters.

if(nargin < 1)
  initialType = 'circle';
end

if(nargin < 2)
  accuracy = 'medium';
end

%---------------------------------------------------------------------------
% Use periodic boundary conditions (usually causes a larger change)?
periodic = true;

% Create the grid.
g.dim = 2;
g.min = -ones(g.dim, 1);
g.dx = 1 / 50;
if(periodic)
  g.max = (1 - g.dx) * ones(g.dim, 1);
  g.bdry = @addGhostPeriodic;
else
  g.max = ones(g.dim, 1);
  g.bdry = @addGhostExtrapolate;
end
g = processGrid(g);

%---------------------------------------------------------------------------
% What kind of display?
if(nargin < 3)
  switch(g.dim)
   case 1
    displayType = 'plot';
   case 2
    displayType = 'surf';
   case 3
    displayType = 'surface';
   otherwise
    error('Default display type undefined for dimension %d', g.dim);
  end
end

%---------------------------------------------------------------------------
% Choose maximum time to reinitialize the right number of grid cells.
%   Of course, there is no point in reinitializing longer than the 
%   diameter of the computational domain.
tMax = min(reinitGridCells * max(g.dx), norm(g.max - g.min));

%---------------------------------------------------------------------------
% Choose the flow field.
switch(initialType)

 case 'circle'
  radius = 0.25 * min(g.max - g.min);
  center = zeros(g.dim, 1) + 0.5 * radius;
  data = shapeSphere(g, center, radius);

 case 'star'
  if(g.dim < 2)
    error('initialType star works only in dimension > 1');
  end
  points = 7;
  shift = 2;
  scale = 0.25;
  data = zeros(size(g.xs{1}));
  [ theta, r ] = cart2pol(g.xs{1}, g.xs{2});
  data = r - scale * (cos(points * theta) + shift);

 otherwise
  error('Unknown initialType %s', initialType);
  
end
data0 = data;

%---------------------------------------------------------------------------
% Initialize Display
f = figure;

h0 = visualizeLevelSet(g, data, displayType, level, 't = 0');

hold on;
if(g.dim > 1)
  axis(g.axis);
  daspect([ 1 1 1 ]);
end

%---------------------------------------------------------------------------
% Reinitialize to signed distance function.
startTime = cputime;
data = signedDistanceIterative(g, data, accuracy, tMax, errorMax);
endTime = cputime;
fprintf('\nTotal execution time %g seconds\n', endTime - startTime);

% Show the results
h = visualizeLevelSet(g, data, displayType, level, [ 't = ' num2str(tMax) ]);

% Compare the magnitude of the gradient.
mag = calculateGradientMagnitude(g, data);
mag0 = calculateGradientMagnitude(g, data0);
figure;
plot(sort(mag0(:)), 'r-');
hold on
plot(sort(mag(:)), 'b-');
legend('Initial Gradient Magnitude', 'Final Gradient Magnitude');



%---------------------------------------------------------------------------
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%---------------------------------------------------------------------------
function mag = calculateGradientMagnitude(grid, data)
% calculateGradientMagnitude: calculates the magnitude of the gradient
%
% Helper function.  Uses Matlab's builtin gradient function, which is mostly
%   centered difference approximations (except on the edge of the domain).

% Approximate the gradient with Matlab's builtin.
dxCell = num2cell(grid.dx);
gs = cell(grid.dim, 1);
[ gs{:} ] = gradient(data, dxCell{:});

% Calculate the magnitude.
mag = zeros(size(data));
for i = 1 : grid.dim
  mag = mag + gs{i}.^2;
end
mag = sqrt(mag);

