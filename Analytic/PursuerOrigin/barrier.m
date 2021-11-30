function [ xp, xm, xc, xpStar, xmStar ] = barrier(theta, beta)
% [ xp, xm, xc, xpStar, xmStar ] = barrier(theta, beta)
%
% computes a collection of points on the barrier of
%    the game of two idenical cars with the PURSUER at the origin
%
% see techrep
% draws on A.W.Merz, "The Game of Two Identical Cars", pp.324 -- 343
%	Journal of Optimization Theory and Applications, 9, 5 (1972)
%
% inputs:
%	theta	relative angle slice to determine crossover at
%	beta	ratio of capture circle to turn radius
%	
% outputs:
%	xp	points on the left barrier
%	xm	points on the right barrier
%	xc	points on the capture circle which are part of the barrier
%	xpStar	key points on the left barrier (between subsets)
%	xmStar	key points on the right barrier (between subsets)
%
% Ian Mitchell, 4/23/01

% argument checking & defaults
if(nargin < 2)
  error('Arguments theta and beta are required');
end
if((theta > 0) | (theta < -pi))
  error('crossover can only handle 0 >= theta >= -pi');
end

% how fine a discretization?
dt = 0.05;
dpsi = dt;

% compute the crossover parameters for this theta
[ bcase, p, pos ] = crossover(theta, beta);

% crossover is an important point
xpStar.crossover = [ pos; theta ];
xmStar.crossover = [ pos; theta ];

% start with left barrier
sigmaE = +1;
left = 1;

% compute the portion of the positive barrier lying on a single trajectory
if(bcase(1) == 2)

  % start with type 3 trajectory, then type 2
  %   all of the type 2 trajectory is on the barrier
  [ z2, z1, z12 ] = optTraj2(3, 2, sigmaE, sigmaE, p.tau1p, p.tau2p, ...
                             [ 0; beta; 0 ], dt);
  xpStar.single = z1;
  % z12 gives points leading to crossover, we want crossover point first
  xp = flipud(z12);
else
  xpStar.single = zeros(3, 0);
  xp = zeros(0,3);
end

if((bcase(1) == 1) | (bcase(1) == 2))
  % the curved portion of the positive barrier
  ts = [ p.tau1p : -dt : 0, 0 ]';
  zs = zeros(length(ts), 3);
  for i = 1:length(ts)
    t1 = ts(i);
    t2 = finalTime(theta, t1, 0, left);
    zs(i, :) = optTraj2(3, 1, sigmaE, sigmaE, t1, t2, [ 0; beta; 0 ])';
  end
  xpStar.curved = zs(end, :)';
  xp = [ xp; zs ];

  % the straight portion of the positive barrier
  psis = [ -dpsi : -dpsi : theta / 2, theta / 2 ];
  zs = zeros(length(psis), 3);
  for i = 1 : length(psis)
    psi = psis(i);
    t2 = finalTime(theta, 0, psi, left);
    z0 = [ beta * sin(psi); beta * cos(psi); 2 * psi ];
    temp = optTraj(1, sigmaE, z0, [ 0:dt:t2, t2 ]');
    zs(i, :) = temp(end, :);
  end
  xpStar.straight = zs(end, :)';
  xp = [ xp; zs ];
else
  xpStar.curved = zeros(3,0);
  xpStar.straight = zeros(3,0);
end

% now right barrier
sigmaE = -1;
left = 0;

% no portion lies on a single trajectory
xmStar.single = zeros(3, 0);
xm = zeros(0, 3);

% the curved portion of the negative barrier
if(bcase(2) == 1)
  ts = [ p.tau1m : -dt : 0, 0 ]';
  zs = zeros(length(ts), 3);
  for i = 1 : length(ts)
    t1 = ts(i);
    t2 = finalTime(theta, t1, 0, left);
    zs(i, :) = optTraj2(3, 1, sigmaE, sigmaE, t1, t2, [ 0; beta; 0 ])';
  end
  xmStar.curved = zs(end, :)';
  xm = [ xm; zs ];  
else
  xmStar.curved = zeros(3,0);
end

% the straight portion of the negative barrier
if((bcase(2) == 1) | (bcase(2) == 2))
  psis = [ p.psi0m : dpsi : pi + theta / 2, pi + theta / 2 ];
  zs = zeros(length(psis), 3);
  for i = 1 : length(psis)
    psi = psis(i);
    t2 = finalTime(theta, 0, psi, left);
    z0 = [ beta * sin(psi); beta * cos(psi); 2 * psi ];
    temp = optTraj(1, sigmaE, z0, [ 0:dt:t2, t2 ]');
    zs(i, :) = temp(end, :);
  end
  xmStar.straight = zs(end, :)';
  xm = [ xm; zs ];
else
  xmStar.straight = zeros(3, 0);
end

% finally, close out the circle
psim = atan(xm(end,2) / xm(end,1));
psip = psim - pi;
psis = [ psip : 3 * dpsi : psim, psim ]';
xc = [ beta * cos(psis), beta * sin(psis), theta * ones(size(psis)) ];


%-------------------------------------------------------------------------
function t2 = finalTime(theta, t1, psi, left)
% computes the time it will take to get from theta(t1) to theta

if((psi ~= 0) & (t1 ~= 0))
  error('Only one of t1 and psi may be nonzero');
end

  if(left)
    t2 = 0.5 * (t1 - theta + 2 * psi);
  else
    t2 = 0.5 * (t1 + 2 * pi + theta - 2 * psi);
  end

