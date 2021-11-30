% script file to generate a figure showing all the trajectories that
%   make up a single slice of the left barrier
%
% not currently set up to handle bcase(1) == 2
%     (ie single trajectory case for theta near zero)
%
% Ian Mitchell, 6/18/01

num_views = 2;

beta = 0.5;
scale = [ 1, 1, 180 / pi ];
aspect = scale .* [ 1 1 2 ];
dtIntraTraj = 0.05;
dtInterTraj = 0.1;
dpsi = dtInterTraj;

figure
set(gcf, 'defaultLineLineWidth', 1, 'defaultLineMarkerSize', 6);

sliceStyle = 'k--';
trajStyle = 'b-';

% Create the plot in the left subfigure.  We will copy it into the other
% subfigures afterwards.
if(num_views > 1)
  subplot(1,num_views,1);
end
hold on;

theta = [ -150; ] * pi / 180;

% plot the barrier (for comparison purposes)
[ xp xm xc sp sm ] = barrier(theta, beta);
xp = xp .* repmat(scale, size(xp, 1), 1);
xm = xm .* repmat(scale, size(xm, 1), 1);
xc = xc .* repmat(scale, size(xc, 1), 1);

% complete the capture circle
xc = [ xc; -xc(:, 1:2), xc(:,3) ];

lhTemp = plot3(xp(:,1), xp(:,2), xp(:,3), sliceStyle, ...
               xm(:,1), xm(:,2), xm(:,3), sliceStyle, ...
               xc(:,1), xc(:,2), xc(:,3), sliceStyle);
lh(1) = lhTemp(1);
hold on;

% put in an additional capture circle at theta == 0
plot3(xc(:,1), xc(:,2), zeros(size(xc(:,3))), sliceStyle);


% get the crossover stats
[ bcase, p, pos ] = crossover(theta, beta);

% left barrier
sigmaE = +1;

% the curved portion of the positive barrier
ts = [ p.tau1p : -dtInterTraj : 0, 0 ]';
for i = 1:length(ts)
  t1 = ts(i);
  t2 = 0.5 * (t1 - theta);
  z0 = [ 0; beta; 0 ];
  [ z2, z1, z12, z01 ] = optTraj2(3, 1, sigmaE,sigmaE,t1,t2,z0,dtIntraTraj);
  z = [ z01; z12 ];
  z = z .* repmat(scale, size(z, 1), 1);
  plot3(z(:,1), z(:,2), z(:,3), trajStyle);
end

% the straight portion of the positive barrier
psis = [ -dpsi : -dpsi : theta / 2, theta / 2 ];
for i = 1 : length(psis)
  psi = psis(i);
  t2 = 0.5 * (- theta + 2 * psi);
  z0 = [ beta * sin(psi); beta * cos(psi); 2 * psi ];
  z = optTraj(1, sigmaE, z0, [ 0:dtIntraTraj:t2, t2 ]');
  z = z .* repmat(scale, size(z, 1), 1);
  plot3(z(:,1), z(:,2), z(:,3), trajStyle);
end

% Copy figure into additional subfigures.
original = gca;
for i = 2:num_views
  subplot(1,num_views,i)
  copyobj(get(original,'Children'),gca);
end

% Make them look pretty.
for i = 1:num_views
  if(num_views > 1)
    subplot(1, num_views, i);
  end

  xlabel('x_1');
  ylabel('x_2');
  zlabel('x_3');
  title([ 'Slice at ' num2str(theta * 180 / pi) '^\circ' ]);
  axis([ -0.75 2 -0.75 2.75 -180 0 ]);
  daspect(aspect);
  grid on;
end

% Choose the views.
if(num_views > 1)
  subplot(1,num_views,1);
  view(2);
  subplot(1,num_views,2);
  view(285,15);
end