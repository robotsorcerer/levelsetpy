function spinAnimation(fig, filename, compress)
% spinAnimation: Create an animation of a spinning 3D plot.
%
%   spinAnimation(fig, filename, compress)
%
% It is difficult to visualize a complex three dimensional isosurface
%   from fixed angles.  This function demonstrates how to use Matlab's
%   animation generation utilities to create a spining animation of a 
%   fixed three dimensional figure.  The animation can be used to liven
%   up overly mathematical talks.
%
% The figure must be generated in advance.
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
% Parameters:
%
%   figure       The figure number holding the plot of interest.
%   filename     The name to give to the animation avi file.
%   compress     Boolean specifying whether to use lossy compression to
%                  (significantly) reduce the file size.  The degree of
%                  compression can be modified by changing the source code.

% Copyright 2004 Ian M. Mitchell (mitchell@cs.ubc.ca).
% This software is used, copied and distributed under the licensing 
%   agreement contained in the file LICENSE in the top directory of 
%   the distribution.
%
% Ian Mitchell, 4/8/03

% Some parameters.
fixElevation = 0;               % Other option: some vertical motion too.
deleteLabels = 1;               % Labels tend to float around and look bad.
frames = 120;
qualityValue = 90;		% Even 100 gets significant compression.

% Turn off stuff we don't want -- animations are typically too busy looking.
if(deleteLabels)
  set(gca, 'XTick', [], 'YTick', [], 'ZTick', []);
  delete(get(gca, 'Title'));
  delete(get(gca, 'XLabel'));
  delete(get(gca, 'YLabel'));
  delete(get(gca, 'ZLabel'));
end

% Set up the figure window nicely (you can change the resolution here).
set(fig, 'Position', [ 100 100 384 384 ], 'color', 'white');
box on

% Set up the view that takes up the largest part of the figure window
%   and freeze that view angle.  Otherwise the figure will appear to
%   zoom in and out as it rotates.
view(45, 35);
drawnow;
camva('manual');

% Set up camera angles for each frame.
az = linspace(0, 360, frames);
az = az([ round(frames/12) : end-1, 1 : round(frames/12) ]);
if(fixElevation)
  el = 15 * ones(size(az));
else
  el = 15 - 20 * sin(az * 2 * pi / 360);
end

% Create the avi file (choose a smaller qualityValue to get smaller files).
if((nargin < 3) | compress)
  mov = avifile(filename, 'quality', qualityValue);
else
  mov = avifile(filename, 'compression', 'none');
end

% Capture the frames.
for i = 1 : frames;
  view(az(i), el(i));
  l = camlight('right');
  drawnow;
  f = getframe(gcf);
  mov = addframe(mov, f);
  delete(l);
end

% We're finished.
mov = close(mov);
