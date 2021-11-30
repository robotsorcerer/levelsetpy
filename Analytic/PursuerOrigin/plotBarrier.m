% script file to create a plot showing a barrier slice
%
% Ian Mitchell 6/18/01

slices = [ -150; -90 ] * pi / 180;
beta = 0.5;

figure
set(gcf, 'defaultLineLineWidth', 1, 'defaultLineMarkerSize', 6);
v = zeros(length(slices), 4);

for i = 1 : length(slices)

  subplot(1,length(slices),i)
  [ xp xm xc sp sm ] = barrier(slices(i), beta);
  hs = plot(sp.crossover(1), sp.crossover(2), 'b*', ...
            xp(:,1), xp(:,2), 'b-', xm(:,1), xm(:,2), 'b--', ...
            xc(:,1), xc(:,2), 'b:', -xc(:,1), -xc(:,2), 'b:');
  axis equal;
  v(i,:) = axis;
end

v = max(v, [], 1);
v = v + 0.1 * sign(v);

for i = 1 : length(slices)
  subplot(1, length(slices), i);

  xlabel('x_1');
  ylabel('x_2');
  title([ 'Slice at ' num2str(slices(i) * 180 / pi) '^\circ' ]);
  axis(v);
  %grid on;
end

legend('crossover point', 'left barrier', 'right barrier', 'capture circle');
