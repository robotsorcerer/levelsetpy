function [ bcase, params, pos ] = crossover(theta, beta)
%  [ bcase, params, pos ] = crossover(theta, beta)
%
% function to determine the crossover point on the barrier of the
%    game of two identical cars with the PURSUER at the origin
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
%	bcase	a 2 x 1 vector
%		  bcase(1) specifies the left crossover trajectory case
%		  bcase(2) specifies the right crossover trajectory case
%	params	structure, will contain values for (unneeded are set to 0)
%		  tau1p, tau1m, tau2p, tau2m, phi0m
%	pos	vector, gives x,y position of crossover point for this theta
%
% Ian Mitchell, 4/23/01

% argument checking & defaults
if(nargin < 2)
  error('Arguments theta and beta are required');
end
if((theta > 0) | (theta < -pi))
  error('crossover can only handle 0 >= theta >= -pi');
end

% optimization options
opts = optimset;

% some initial condition guesses for the various cases
x01 = [ 2.4; 2.3 ];
x02 = [ 2.3; 1.6 ];
x03 = [ 1.7; 1 ];

% first, assume case 1 on left and right
bcase = [ 1; 1 ];
[ tau fval flag ] = fsolve('equality', x01, opts, bcase, theta, beta);
p.tau2p = tau(1);
p.tau2m = tau(2);
p.tau1p = 2 * p.tau2p + theta;
p.tau1m = 2 * p.tau2m - (2 * pi + theta);
p.psi0m = 0;

if(p.tau1m < 0)
  % try case 2 on the right
  bcase = [ 1; 2 ];
  [ tau fval flag ] = fsolve('equality', x02, opts, bcase, theta, beta);
  p.tau2p = tau(1);
  p.tau2m = tau(2);
  p.tau1p = 2 * p.tau2p + theta;
  p.tau1m = 0;
  p.psi0m = pi + theta / 2 - p.tau2m;
elseif(p.tau1p > p.tau2p)
  % try case 2 on the left
  bcase = [ 2; 1 ];
  [ tau fval flag ] = fsolve('equality', x02, opts, bcase, theta, beta);
  p.tau2p = tau(1);
  p.tau2m = tau(2);
  p.tau1p = -theta;
  p.tau1m = 2 * p.tau2m - (2 * pi + theta);
  p.psi0m = 0;
end

if(p.tau1p > p.tau2p)
  % try case 2 on left and right
  bcase = [ 2; 2 ];
  [ tau fval flag ] = fsolve('equality', x03, opts, bcase, theta, beta);
  p.tau2p = tau(1);
  p.tau2m = tau(2);
  p.tau1p = -theta;
  p.tau1m = 0;
  p.psi0m = pi + theta / 2 - p.tau2m;
end

if(nargout > 2)
  % determine the crossover point
  x = crossX(bcase, theta, beta, 1, p.tau2p);
  y = crossY(bcase, theta, beta, 1, p.tau2p);
  pos = [ x; y ];
end

% detect some potential bugs in solution
if((p.tau2p < 0) | (p.tau2m < 0) | (p.tau1p < 0) | (p.tau1m < 0))
  warning('one of the optimal times was negative');
end

if((p.tau1p > p.tau2p) | (p.tau1m > p.tau2m))
  warning('one of the switch tau is greater than its corresponding end tau');
end

if((p.tau1m ~= 0) & (p.psi0m ~= 0))
  warning('tau1m and psi0m are not allowed to both be nonzero');
end

if(flag < 0)
  err = equality([ p.tau2p; p.tau2m ], bcase, theta, beta);
  warning([ 'may not have found a true crossover point; error is ' ...
            num2str(norm(err)) ]);
end

params = p;
