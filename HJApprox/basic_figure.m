% Script file to generate the visualization of the RCBRT for the game of
% two identical Dubins cars.

% Created by Ian Mitchell, 2020-05-18

% Create the model object for the basic problem.  Use the default settings
% for everything.
model = BasicModel();

% Calculate the RCBRT, and capture the grid and target set as well.  This
% function will take a while to run, and will generate a figure while it
% runs.
[ rcbrt, g, target ] = compute_rcbrt(model);

% Create new figure to show target set and RCBRT.
figure;
subplot(1,2,1);
h_target = visualizeLevelSet(g, target, 'surface', 0);
hold on;
h_rcbrt = visualizeLevelSet(g, rcbrt, 'surface', 0);
% Make it prettier.
set(h_target, 'FaceColor', 'blue');
set(h_rcbrt, 'FaceColor', 'red', 'FaceAlpha', 0.5);
daspect([ 1 1 2 ]);
view(120,10);
camlight right;
axis(g.axis);
grid on;
xlabel('x_1');  ylabel('x_2');  zlabel('x_3');

% Create new figure to show RCBRT by itself.
subplot(1,2,2);
h_rcbrt2 = visualizeLevelSet(g, rcbrt, 'surface', 0);
% Make it prettier.
set(h_rcbrt2, 'FaceColor', 'red');
daspect([ 1 1 2 ]);
view(60,10);
camlight headlight;
axis(g.axis);
grid on;
xlabel('x_1');  ylabel('x_2');  zlabel('x_3');
