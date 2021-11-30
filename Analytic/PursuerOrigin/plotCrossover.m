% script file to create a plot of the crossover trajectories
%
% a good viewing angle is 80, 20
%
% Ian Mitchell, 6/18/01

slices = [ -150; -90 ] * pi / 180;
beta = 0.5;
scale = [ 1, 1, 180 / pi ];
aspect = scale .* [ 1 1 2 ];
dt = 0.05;

figure
set(gcf, 'defaultLineLineWidth', 1, 'defaultLineMarkerSize', 6);

for i = 1 : length(slices)

  subplot(1,length(slices),i)

  % plot the barrier (for comparison purposes)
  [ xp xm xc sp sm ] = barrier(slices(i), beta);
  xp = xp .* repmat(scale, size(xp, 1), 1);
  xm = xm .* repmat(scale, size(xm, 1), 1);
  xc = xc .* repmat(scale, size(xc, 1), 1);

  % complete the capture circle
  xc = [ xc; -xc(:, 1:2), xc(:,3) ];

  lhTemp = plot3(xp(:,1), xp(:,2), xp(:,3), 'm:', ...
                 xm(:,1), xm(:,2), xm(:,3), 'm:', ...
                 xc(:,1), xc(:,2), xc(:,3), 'm:');
  lh(1) = lhTemp(1);
  hold on;

  % get the crossover stats
  [ bcase, p, pos ] = crossover(slices(i), beta);

  % left barrier
  sigmaE = +1;

  z0 = [ 0; beta; 0 ];
  if(bcase(1) == 1)
    [ z2, z1, z12, z01 ] = optTraj2(3, 1, sigmaE, sigmaE, p.tau1p, p.tau2p, ...
                                    z0, dt);

  else
    % start with type 3 trajectory, then type 2
    [ z2, z1, z12, z01 ] = optTraj2(3, 2, sigmaE, sigmaE, p.tau1p, p.tau2p, ...
                                    z0, dt);
  end
  z = [ z01; z12 ];
  z = z .* repmat(scale, size(z, 1), 1);
  z0 = z0 .* scale';  z1 = z1 .* scale';  z2 = z2 .* scale';
  lhTemp = plot3(z(:,1), z(:,2), z(:,3), 'b-', z0(1), z0(2), z0(3), 'bo', ...
                 z1(1), z1(2), z1(3), 'bs', z2(1), z2(2), z2(3), 'b*');
  lh(2) = lhTemp(1);

  % add a capture circle at z0
  plot3(xc(:,1), xc(:,2), z0(3) * ones(size(xc(:,3))), 'm:');

  % right barrier
  sigmaE = -1;

  if(bcase(2) == 1)
    z0 = [ 0; beta; 0 ];
    [ z2, z1, z12, z01 ] = optTraj2(3, 1, sigmaE, sigmaE, p.tau1m, p.tau2m, ...
                                    z0, dt);
    z = [ z01; z12 ];

  else
    z0 = [ beta * sin(p.psi0m); beta * cos(p.psi0m); 2 * p.psi0m ];
    z = optTraj(1, sigmaE, z0, [ 0:dt:p.tau2m, p.tau2m ]');
    z1 = z0;
    z2 = z(end,:)';
  end
  z = z .* repmat(scale, size(z, 1), 1);
  z0 = z0 .* scale';  z1 = z1 .* scale';  z2 = z2 .* scale';

  % to avoid jump in trajectory, need to separate z out
  zp = z(find(z(:,3) >= 0), :);
  zm = z(find(z(:,3) < 0), :);

  lhTemp = plot3(zp(:,1), zp(:,2), zp(:,3), 'r-.', ...
                 zm(:,1), zm(:,2), zm(:,3), 'r-.', ...
                 z0(1), z0(2), z0(3), 'ro', ...
                 z1(1), z1(2), z1(3), 'rs', z2(1), z2(2), z2(3), 'r*');
  lh(3) = lhTemp(1);

  % add a capture circle at z0
  plot3(xc(:,1), xc(:,2), z0(3) * ones(size(xc(:,3))), 'm:');

end

for i = 1 : length(slices)
  subplot(1, length(slices), i);

  xlabel('x_1');
  ylabel('x_2');
  zlabel('x_3');
  title([ 'Slice at ' num2str(slices(i) * 180 / pi) '^\circ' ]);
  axis([ -0.75 2 -0.75 2.75 -180 180 ]);
  daspect(aspect);
  grid on;
  view(285,15);
end

legend(lh, { 'barrier', 'left trajectory', 'right trajectory' });
