function xout = equality(x, bcase, theta, beta)
%
% helper function for crossover.m, called by fsolve
% equates the left and right trajectories
%
% Ian Mitchell, 4/23/01

tau2p = x(1);
tau2m = x(2);

xout = zeros(2,1);
xout(1) = crossX(bcase, theta, beta, 1, tau2p) ...
          - crossX(bcase, theta, beta, 0, tau2m);
xout(2) = crossY(bcase, theta, beta, 1, tau2p) ...
	  - crossY(bcase, theta, beta, 0, tau2m);

