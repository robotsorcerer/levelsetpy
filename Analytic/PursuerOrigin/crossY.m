function y = crossY(bcase, theta, beta, left, t)
%
% computes the crossover point for the given tau = t
%    for the given crossover case with the PURSUER at the origin
%    if left = 1 computes the left trajectory, otherwise the right
% helper function for crossover.m
%
% Ian Mitchell, 4/23/01

  if(left)					% left trajectory
    switch bcase(1)
    case 1
      y = -2 * sin(theta + t) + sin(theta) ...
	      + (beta + theta + 2 * t) * cos(theta + t);
    case 2
      y = sin(theta) + (beta - theta) * cos(t + theta);
    otherwise
      error([ 'case ' num2str(bcase(1)) ' is not supported on left' ]);
    end
  else						% right trajectory
    switch bcase(2)
    case 1
      y = 2 * sin(theta - t) - sin(theta) + ...
	      + (beta + 2 * t - (2 * pi + theta)) * cos(theta - t);
    case 2
      y = sin(t) + beta * cos(pi + theta / 2) ...
              + sin(theta - t) - sin(theta);
    otherwise
      error([ 'case ' num2str(bcase(2)) ' is not supported on right' ]);
    end
  end
