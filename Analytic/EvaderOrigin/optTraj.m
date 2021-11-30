function zs = optTraj(type, sigmaE, z0, ts)
% zs = optTraj(type, sigmaE, z0, ts)
%
% computes points on an optimal trajectory for
%     the game of two identical cars with the EVADER at the origin
%
% see techrep
% draws on A.W.Merz, "The Game of Two Identical Cars", pp.324 -- 343
%	Journal of Optimization Theory and Applications, 9, 5 (1972)
%
% inputs:
%	type	{ 1, 2, 3 } type of trajectory
%		  type 1, sigmaP = -sigmaE
%		  type 2, sigmaP = sigmaE
%		  type 3, sigmaP = 0
%	sigmaE	evader's input (+-1)
%	z0	vector of initial conditions (x, y, theta)
%	ts	times at which to compute points
%	
% outputs:
%	zs	vector of points on optimal trajectory
%
% Ian Mitchell, 4/23/01

% argument checking
if(nargin < 4)
  error('Arguments type, sigmaE, z0, and ts are required');
end
if(abs(sigmaE) ~= 1)
  error('evader''s input must be +- 1');
end
if(length(z0) ~= 3)
  error('initial conditions z0 = [ x0; y0; theta0 ]');
end
if(size(ts, 2) ~= 1);
  error('time sequence ts must be given as a column vector');
end

  switch type
  case 1
    zs = type1(sigmaE, z0, ts);
  case 2
    zs = type2(sigmaE, z0, ts);
  case 3
    zs = type3(sigmaE, z0, ts);
  otherwise
    error('type = {1, 2, 3} are only trajectory types allowed');
  end

% handle wrap of theta > pi
zs(:,3) = zs(:,3) - 2 * pi * (zs(:,3) > pi - sqrt(eps));

%----------------------------------------------------------------------
function zs = type1(sigma, z0, ts);
% computes trajectories where pursuer and evader turn opposite directions

x0 = z0(1);
y0 = z0(2);
theta0 = z0(3);

thetas = theta0 + 2 * sigma * ts;
xs = x0 * cos(ts) ...
        + sigma * (1 - cos(ts) + y0 * sin(ts) + cos(thetas) ...
		     - cos(theta0 + sigma * ts));
ys = y0 * cos(ts) + sin(ts) ...
        - sigma * (x0 * sin(ts) + sin(thetas) - sin(theta0 + sigma * ts));

zs = [ xs ys thetas ];

%----------------------------------------------------------------------
function zs = type2(sigma, z0, ts);
% computes trajectories where pursuer and evader turn the same direction

x0 = z0(1);
y0 = z0(2);
theta0 = z0(3);

thetas = theta0 * ones(size(ts));
xs = x0 * cos(ts) ...
        + sigma * (1 - cos(ts) + y0 * sin(ts) - cos(theta0) ...
		     + cos(theta0 + sigma * ts));
ys = y0 * cos(ts) + sin(ts) ...
        - sigma * (x0 * sin(ts) - sin(theta0) + sin(theta0 + sigma * ts));

zs = [ xs ys thetas ];

%----------------------------------------------------------------------
function zs = type3(sigma, z0, ts);
% computes trajectories where pursuer does not turn

x0 = z0(1);
y0 = z0(2);
theta0 = z0(3);

thetas = theta0 + sigma * ts;
xs = x0 * cos(ts) - ts .* sin(thetas) ...
	+ sigma * (1 - cos(ts) + y0 * sin(ts));
ys = y0 * cos(ts) - ts .* cos(thetas) + sin(ts) - sigma * x0 * sin(ts);
zs = [ xs ys thetas ];
