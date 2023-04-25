function [ data, g, data0 ] = flock(accuracy)

% Integration parameters.
tMax = 2.8;                  % End time.
plotSteps = 9;               % How many intermediate plots to produce?
t0 = 0;                      % Start time.
singleStep = 0;              % Plot at each timestep (overrides tPlot).

% Period at which intermediate plots should be produced.
tPlot = (tMax - t0) / (plotSteps - 1);

% How close (relative) do we need to get to tMax to be considered finished?
small = 100 * eps;

% What kind of dissipation?
dissType = 'global';

% Problem Parameters.
%   targetRadius  Radius of target circle (positive).
%   velocityA	  Speed of the evader (positive constant).
%   velocityB	  Speed of the pursuer (positive constant).
%   inputA	  Maximum turn rate of the evader (positive).
%   inputB	  Maximum turn rate of the pursuer (positive).
targetRadius = 5;
velocityA = 5;
velocityB = 5;
inputA = 1;
inputB = 1;

%---------------------------------------------------------------------------
% What level set should we view?
level = 0;

% Visualize the 3D reachable set.
displayType = 'surface';

% Pause after each plot?
pauseAfterPlot = 0;

% Delete previous plot before showing next?
deleteLastPlot = 1;

% Visualize the angular dimension a little bigger.
aspectRatio = [ 1 1 0.4 ];

% Plot in separate subplots (set deleteLastPlot = 0 in this case)?
useSubplots = 0;

%---------------------------------------------------------------------------
% Approximately how many grid cells?
%   (Slightly different grid cell counts will be chosen for each dimension.)
Nx = 51;

% Create the grid.
g.dim = 3;
g.min = [  -1.5; 1.5;     -pi];
g.max = [ 1.5; 1.5; pi ];

g.bdry = { @addGhostExtrapolate; @addGhostExtrapolate; @addGhostPeriodic };
% Roughly equal dx in x and y (so different N).
g.N = [ Nx; ceil(Nx * (g.max(2) - g.min(2)) / (g.max(1) - g.min(1))); Nx-1 ];
% Need to trim max bound in \psi (since the BC are periodic in this dimension).
g.max(3) = g.max(3) * (1 - 1 / g.N(3));

g.grid = createGrid(g.min, g.max, g.N, 3, true);
g = processGrid(g);

%---------------------------------------------------------------------------
% Create initial conditions (cylinder centered on origin).
data = shapeCylinder(g, 3, [ 0; 0; 0 ], targetRadius);
data0 = data;