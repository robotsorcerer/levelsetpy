function [ data, g, data0 ] = animateAir3D(filename, accuracy, compress)
% animateAir3D: create an animation of the growth of the air3D reach set.
%
%   [ data, g, data0 ] = animateAir3D(filename, accuracy, compress)
%  
% This file generates an animation showing how the reachable set grows
% as time progresses.  It is basically a combination of the files:
%   air3D (which sets up and handles the reach set calculation),
%   spinAnimation (which has the code necessary for animations).
%
% NOTES:
%
% 1) This function will probably only work in Windows, since it uses
%    the avifile command and the avi animation format.
%
% 2) While the animation is being generated, don't move anything
%    (mouse, other windows) in front of the matlab figure window,
%    otherwise that image will be captured into the avi.
%
% 3) If this file stops because of an error, the avi file will remain open.
%    In that case, you must issue a "clear all" command to get rid of
%    the open file handle and then delete the partially completed avi file.
%
%
% In this example, the target set is a circle at the origin (cylinder in 3D)
% that represents a collision in relative coordinates between the evader
% (player a, fixed at the origin facing right) and the pursuer (player b).
%
% The relative coordinate dynamics are
%
%   \dot x    = -v_a + v_b \cos \psi + a y
%	  \dot y    = v_b \sin \psi - a x
%	  \dot \psi = b - a
%
% where v_a and v_b are constants, input a is trying to avoid the target
%	input b is trying to hit the target.
%
% For more details, see my PhD thesis, section 3.1.
%
% This function was originally designed as a script file, so most of the
% options can only be modified in the file.  For example, edit the file to
% change the grid dimension, boundary conditions, aircraft parameters, etc.
%
% To get exactly the result from the thesis choose:
%   targetRadius = 5, velocityA = velocityB = 5, inputA = inputB = +1.
%
% It is also possible to modify the file to choose the camera motion
% as the reach set grows.
%
% Input Parameters:
%
%   filename: String.  The name to give to the animation avi file.
%
%   accuracy: Controls the order of approximations.  Options are:
%                  'low'         Use odeCFL1 and upwindFirstFirst.
%                  'medium'      Use odeCFL2 and upwindFirstENO2 (default).
%                  'high'        Use odeCFL3 and upwindFirstENO3.
%                  'veryHigh'    Use odeCFL3 and upwindFirstWENO5.
%
%   compress: Boolean specifying whether to use lossy compression to
%   (significantly) reduce the file size.  The degree of compression can be
%   modified by changing the source code.  Optional.  Default is true.
%
% Ouput Parameters:
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
% Ian Mitchell, 7/14/04
% Subversion tags for version control purposes.
% $Date: 2011-03-29 21:49:03 -0700 (Tue, 29 Mar 2011) $
% $Id: animateAir3D.m 61 2011-03-30 04:49:03Z mitchell $

%---------------------------------------------------------------------------
% Make sure we can see the kernel m-files.
run('setup.m');

%---------------------------------------------------------------------------
% You will see many executable lines that are commented out.
%   These are included to show some of the options available; modify
%   the commenting to modify the behavior.

% Optional input parameters.
close all
if(nargin < 2)
  accuracy = 'medium';
end
if(nargin < 3)
  compress = 1;
end

%---------------------------------------------------------------------------
% Some animation parameters.
fixViewAngle = 1;               % Other option: camera rotates around set.
fixElevation = 1;               % Other option: some vertical motion too.
deleteLabels = 1;               % Labels tend to float around and look bad.
frames = 131;
qualityValue = 90;		% Even 100 gets significant compression.
fixedView = [ 15, 15 ];         % If you are using a fixed view angle.

%---------------------------------------------------------------------------
% Integration parameters.
tMax = 2.6;                  % End time.
t0 = 0;                      % Start time.
singleStep = 0;              % Plot at each timestep (overrides tPlot).

% Period at which intermediate plots should be produced.
tPlot = (tMax - t0) / (frames - 1);

% How close (relative) do we need to get to tMax to be considered finished?
small = 100 * eps;

% What kind of dissipation?
dissType = 'global';

%---------------------------------------------------------------------------
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

%---------------------------------------------------------------------------
% Approximately how many grid cells?
%   (Slightly different grid cell counts will be chosen for each dimension.)
Nx = 51;

% Create the grid.
g.dim = 3;
g.min = [  -6; -10;     0 ];
g.max = [ +20; +10; +2*pi ];
g.bdry = { @addGhostExtrapolate; @addGhostExtrapolate; @addGhostPeriodic };
% Roughly equal dx in x and y (so different N).
g.N = [ Nx; ceil(Nx * (g.max(2) - g.min(2)) / (g.max(1) - g.min(1))); Nx-1 ];
% Need to trim max bound in \psi (since the BC are periodic in this dimension).
g.max(3) = g.max(3) * (1 - 1 / g.N(3));
g = processGrid(g);

%---------------------------------------------------------------------------
% Create initial conditions (cylinder centered on origin).
data = shapeCylinder(g, 3, [ 0; 0; 0 ], targetRadius);
data0 = data;

%---------------------------------------------------------------------------
% Set up spatial approximation scheme.
schemeFunc = @termLaxFriedrichs;
schemeData.hamFunc = @air3DHamFunc;
schemeData.partialFunc = @air3DPartialFunc;
schemeData.grid = g;

% The Hamiltonian and partial functions need problem parameters.
schemeData.velocityA = velocityA;
schemeData.velocityB = velocityB;
schemeData.inputA = inputA;
schemeData.inputB = inputB;

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
% Initialize Display
fig = figure;
h = visualizeLevelSet(g, data, displayType, level, [ 't = ' num2str(t0) ]);
axis(g.axis);

%---------------------------------------------------------------------------
% Now establish the animation stuff.

% Set up the figure window nicely (you can change the resolution here).
set(fig, 'Position', [ 100 100 384 384 ], 'color', 'white');
box on

% Turn off stuff we don't want -- animations are typically too busy looking.
if(deleteLabels)
  set(gca, 'XTick', [], 'YTick', [], 'ZTick', []);
  delete(get(gca, 'Title'));
  delete(get(gca, 'XLabel'));
  delete(get(gca, 'YLabel'));
  delete(get(gca, 'ZLabel'));
end

% Set up camera angles for each frame.
if(fixViewAngle)
  view(fixedView)
  az = fixedView(1) * ones(frames, 1);
  el = fixedView(2) * ones(frames, 1);
else
  % Set up the view that takes up the largest part of the figure window
  %   and freeze that view angle.  Otherwise the figure will appear to
  %   zoom in and out as it rotates.
  az = linspace(0, 360, frames);
  az = az([ round(frames/12) : end-1, 1 : round(frames/12) ]);

  if(fixElevation)
    view(45, fixedView(2));
    el = fixedView(2) * ones(size(az));
  else
    view(45,45);
    el = 30 + 20 * sin(az * 2 * pi / 360);
  end
end
% Visualize the angular dimension a little bigger.
aspectRatio = [ 1 1 0.4 ];
daspect(aspectRatio);

% Freeze the view angle, so that figure will not zoom in and out.
camva('manual');

% Create the avi file (choose a smaller qualityValue to get smaller files).
% if compress
%   mov = avifile(filename, 'quality', qualityValue);
% else
%   mov = avifile(filename, 'compression', 'none');
% end
vout = VideoWriter('air3D_anim.mp4', 'MPEG-4');
vout.Quality = 100;

vout.FrameRate = 30;

try
  vout.open;
catch
   error('cannot open file for writing')
end

%---------------------------------------------------------------------------
% Capture the first frame.

frame = 1;
view(az(frame), el(frame));
l = camlight('right');
drawnow;
f = getframe(gcf);
% mov = addframe(mov, f);
writeVideo(vout,f);
delete(l);

%---------------------------------------------------------------------------
% Loop until tMax (subject to a little roundoff).
tNow = t0;
startTime = cputime;
while(tMax - tNow > small * tMax)

  % Reshape data array into column vector for ode solver call.
  y0 = data(:);

  % How far to step?
  tSpan = [ tNow, min(tMax, tNow + tPlot) ];
  
  % Take a timestep.
  [ t y ] = feval(integratorFunc, schemeFunc, tSpan, y0,...
                  integratorOptions, schemeData);
  tNow = t(end);

  % Get back the correctly shaped data array
  data = reshape(y, g.shape);

  % Delete last visualization if necessary.
  delete(h);

  % Create new visualization.
  h = visualizeLevelSet(g, data, displayType, level, [ 't = ' num2str(tNow) ]);
  axis(g.axis);

  % Get next frame.
  frame = frame + 1;
  view(az(frame), el(frame));
  l = camlight('right');
  drawnow;
  f = getframe(gcf);
  %mov = addframe(mov, f);
  writeVideo(vout,f);
  delete(l);

end

endTime = cputime;
fprintf('Total execution time %g seconds', endTime - startTime);

% We're finished with the movie.
vout.close; %#ok<NASGU>



%---------------------------------------------------------------------------
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%---------------------------------------------------------------------------
function hamValue = air3DHamFunc(~, ~, deriv, schemeData)
% air3DHamFunc: analytic Hamiltonian for 3D collision avoidance example.
%
% hamValue = air3DHamFunc(t, data, deriv, schemeData)
%
% This function implements the hamFunc prototype for the three dimensional
%   aircraft collision avoidance example (also called the game of
%   two identical vehicles).
%
% It calculates the analytic Hamiltonian for such a flow field.
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
%   .velocityA	 Speed of the evader (positive constant).
%   .velocityB	 Speed of the pursuer (positive constant).
%   .inputA	 Maximum turn rate of the evader (positive).
%   .inputB	 Maximum turn rate of the pursuer (positive).
%
% Ian Mitchell 3/26/04

checkStructureFields(schemeData, 'grid', 'velocityA', 'velocityB', ...
                                 'inputA', 'inputB');

grid = schemeData.grid;

% implements equation (3.3) from my thesis term by term
%   with allowances for \script A and \script B \neq [ -1, +1 ]
%   where deriv{i} is p_i
%         x_r is grid.xs{1}, y_r is grid.xs{2}, \psi_r is grid.xs{3}
%         v_a is velocityA, v_b is velocityB, 
%         \script A is inputA and \script B is inputB
hamValue = -(-schemeData.velocityA * deriv{1} ...
	     + schemeData.velocityB * cos(grid.xs{3}) .* deriv{1} ...
	     + schemeData.velocityB * sin(grid.xs{3}) .* deriv{2} ...
	     + schemeData.inputA * abs(grid.xs{2} .* deriv{1} ...
                                    - grid.xs{1} .* deriv{2} - deriv{3})...
	     - schemeData.inputB * abs(deriv{3}));



%---------------------------------------------------------------------------
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%---------------------------------------------------------------------------
function alpha = air3DPartialFunc(~, ~, ~, ~, schemeData, dim)
% air3DPartialFunc: Hamiltonian partial fcn for 3D collision avoidance example.
%
% alpha = air3DPartialFunc(t, data, derivMin, derivMax, schemeData, dim)
%
% This function implements the partialFunc prototype for the three dimensional
%   aircraft collision avoidance example (also called the game of
%   two identical vehicles).
%
% It calculates the extrema of the absolute value of the partials of the 
%   analytic Hamiltonian with respect to the costate (gradient).
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
%   .velocityA	 Speed of the evader (positive constant).
%   .velocityB	 Speed of the pursuer (positive constant).
%   .inputA	 Maximum turn rate of the evader (positive).
%   .inputB	 Maximum turn rate of the pursuer (positive).
%
% Ian Mitchell 3/26/04

checkStructureFields(schemeData, 'grid', 'velocityA', 'velocityB', ...
                                 'inputA', 'inputB');

grid = schemeData.grid;

switch dim
  case 1
    alpha = abs(-schemeData.velocityA + ...
                + schemeData.velocityB * cos(grid.xs{3})) ...
            + schemeData.inputA * abs(grid.xs{2});

  case 2
    alpha = abs(schemeData.velocityB * sin(grid.xs{3})) ...
            + schemeData.inputA * abs(grid.xs{1});

  case 3
    alpha = schemeData.inputA + schemeData.inputB;

  otherwise
    error([ 'Partials for the game of two identical vehicles' ...
            ' only exist in dimensions 1-3' ]);
end
