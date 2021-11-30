function zdot = trajODE(t, z, flag, sigma)
% zdot = trajODE(t, z, sigma)
%
% implements the optimal trajectories for the game of two identical cars
%	equations (2)
%
% draws on A.W.Merz, "The Game of Two Identical Cars", pp.324 -- 343
%	Journal of Optimization Theory and Applications, 9, 5 (1972)
%
% Ian Mitchell for HS'01, 1/16/01

reverseTime = 1;

sigma1 = sigma(1);
sigma2 = sigma(2);

x = z(1);
y = z(2);
theta = z(3);

xdot = -sigma1 * y + sin(theta);
ydot = -1 + sigma1 * x + cos(theta);
thetadot = -sigma1 + sigma2;

zdot = (reverseTime * -1) * [ xdot; ydot; thetadot ];
