function x = crossX(bcase, theta, beta, left, t)
%
% computes the crossover point for the given tau = t
%    for the given crossover case with the EVADER at the origin
%    if left = 1 computes the left trajectory, otherwise the right
% helper function for crossover.m
% 
% Ian Mitchell, 4/23/01

  if(left)					% left trajectory
    switch bcase(1)
    case 1
      x = 1 - 2 * cos(t) + cos(theta) ...
              - (beta + 2 * t - 2 * pi - theta) * sin(t);
    case 2
      x = 1 - cos(t) - beta * sin(pi + theta / 2) - cos(theta - t) +cos(theta);
    otherwise
      error([ 'case ' num2str(bcase(1)) ' is not supported on left' ]);
    end
  else						% right trajectory
    switch bcase(2)
    case 1
      x = 2 * cos(t) - 1 - cos(theta) + (beta + 2 * t + theta) * sin(t);
    case 2
      x = -1 + cos(theta) + (beta - theta) * sin(t);
    otherwise
      error([ 'case ' num2str(bcase(2)) ' is not supported on right' ]);
    end
  end
