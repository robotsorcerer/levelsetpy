% Script file to generate the visualization of the RCBRT for the game of
% two identical Dubins cars for the no heading case.

% Created by Ian Mitchell, 2020-05-18

% Do you want to include contours of the analytic solution?
include_analytic = true;

% Create the model object for the basic problem.  Use the default settings
% for everything.
model = NoHeadingModel();

% Calculate the RCBRT, and capture the grid and target set as well.  This
% function will take a while to run, and will generate a figure while it
% runs.
[ rcbrt, g, target ] = compute_rcbrt(model);

% Create new figure to show target set and RCBRT.
figure;
hold on;

% Draw the analytic solution in first if necessary, so it appears behind
% the HJI solution.
if(include_analytic)
  slices = [ -180; -150; -120; -90; -60; -30; 0] * pi / 180;
  beta = 0.5;

  colors = { 'm'; 'm'; 'm' };
  styles = { '--'; '--'; '--' };

  set(gcf, 'defaultLineLineWidth', 1, 'defaultLineMarkerSize', 6);

  for i = 1:length(slices)
    [ xp, xm, xc ] = barrier(slices(i), beta);
    plot(xp(:,2), xp(:,1), [ colors{1}, styles{1} ], ...
      xm(:,2), xm(:,1), [ colors{2}, styles{2} ], ...
      xc(:,2), xc(:,1), [ colors{3}, styles{3} ]);
  end
end

h_target = visualizeLevelSet(g, target, 'contour', 0);
h_rcbrt = visualizeLevelSet(g, rcbrt, 'contour', 0);
set(h_target, 'LineColor', 'blue', 'LineWidth', 2, 'LineStyle', ':');
set(h_rcbrt, 'LineColor', 'red', 'LineWidth', 2, 'LineStyle', '-')
daspect([ 1 1 2 ]);
axis(g.axis);
grid on;
xlabel('x_1');  ylabel('x_2');

