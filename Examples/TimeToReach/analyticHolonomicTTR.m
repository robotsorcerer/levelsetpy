function mttr = analyticHolonomicTTR(whichNorm, gridIn)
% analyticHolonomicTTR: analytic solution, holonomic time to reach example.
%
%   mttr = analyticHolonomicTTR(whichNorm, grid)
%  
% Computes the analytic minimum time to reach each node in the grid
%   for the holonomic 2D integrator under unit bounded input.
%
% The dynamics are
%
%        \dot x    = b_1
%	 \dot y    = b_2
%
%   where input ||b|| \leq 1 is trying to hit the target.  The norm
%   in which b is bounded can be modified to produce different shapes
%   of time to reach function.
%
% The true solution for point x is simply ||x|| in whichever norm
%   the input is bounded.
%
% Parameters:
%
%   whichNorm    Controls the norm in which b is bounded.
%                  '1',1,'sum'       norm(b, 1) = sum(abs(b)) <= 1.
%                  '2',2,'rms'       norm(b, 2) = sqrt(sum(b.^2)) <= 1.
%                  'inf',inf,'max'   norm(b, inf) = max(abs(b)) <= 1.
%   grid         Grid structure on which data is to be computed.
%
%   mttr         Minimum time to reach function.

% Copyright 2004 Ian M. Mitchell (mitchell@cs.ubc.ca).
% This software is used, copied and distributed under the licensing 
%   agreement contained in the file LICENSE in the top directory of 
%   the distribution.
%
% Ian Mitchell, 12/06/04

run('setup.m');
switch(whichNorm)

 case { '1'; 1; 'sum' }
  mttr = abs(gridIn.xs{1}) + abs(gridIn.xs{2});

 case { '2'; 2; 'rms' }
  mttr = sqrt(gridIn.xs{1}.^2 + gridIn.xs{2}.^2);

 case { 'inf'; inf; 'max' }
  mttr = max(abs(gridIn.xs{1}), abs(gridIn.xs{2}));

 otherwise
  error('Unknown type of norm: %s', schemeData.whichNorm);

end
