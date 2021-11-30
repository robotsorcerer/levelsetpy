% script file to recreate figure 7 from
%   A.W.Merz, "The Game of Two Identical Cars", pp.324 -- 343
%	Journal of Optimization Theory and Applications, 9, 5 (1972)
%
% Ian Mitchell, 1/21/01

num_views = 2;

slices = [ -180; -150; -120; -90; -60; -30; 0] * pi / 180;
beta = 0.5;
scale = [ 1, 1, 180 / pi ];
aspect = scale;

closeCircle = 1;
closeCircleStyle = 'b--';

colors = { 'b'; 'b'; 'b' };
styles = { '-'; '-'; '-' };
%styles = { '-'; '--'; '-.' };
%colors = { 'b'; 'r'; 'm' };
%styles = { 'x-'; 's-'; 'o-' };

plotSpecial = true;
specialStyles = { '*', 's', 'v', 'o' };
specialNames = { 'crossover', 'single', 'curved', 'straight' };

if(closeCircle)
  cx = beta * scale(1) * cos(0 : pi/90 : 2*pi);
  cy = beta * scale(2) * sin(0 : pi/90 : 2*pi);
end

figure;
set(gcf, 'defaultLineLineWidth', 1, 'defaultLineMarkerSize', 6);
if(num_views > 1)
  % Create visualization in first subplot, copy to the rest.
  subplot(1, num_views, 1);
end
hold on;

for i = 1:length(slices)

  [ xp xm xc sp sm ] = barrier(slices(i), beta);
  scalep = repmat(scale, size(xp, 1), 1);
  scalem = repmat(scale, size(xm, 1), 1);
  scalec = repmat(scale, size(xc, 1), 1);
  xp = xp .* scalep;  xm = xm .* scalem;  xc = xc .* scalec;
  plot3(xp(:,1), xp(:,2), xp(:,3), [ colors{1}, styles{1} ], ...
	xm(:,1), xm(:,2), xm(:,3), [ colors{2}, styles{2} ], ...
	xc(:,1), xc(:,2), xc(:,3), [ colors{3}, styles{3} ]);

  if(plotSpecial)
    pts = { sp.crossover; sp.single; sp.curved; sp.straight; ...
            sm.crossover; sm.single; sm.curved; sm.straight };
    for j = 1 : length(pts)
      pt = repmat(scale', 1, size(pts{j}, 2)) .* pts{j};
      color = colors{fix((j - 1) / length(specialNames)) + 1};
      style = specialStyles{rem((j - 1), length(specialNames)) + 1};
      if(~isempty(pt))
        plot3(pt(1), pt(2), pt(3), [ color, style ]);
      end
    end
  end

  if(closeCircle)
    cz = xc(1,3) * ones(size(cx));
    plot3(cx, cy, cz, closeCircleStyle);
  end
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
  axis([ -0.75 1.5 -0.75 3.0 -180 0 ]);
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
