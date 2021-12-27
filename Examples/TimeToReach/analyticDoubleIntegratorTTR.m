function mttr = analyticDoubleIntegratorTTR(grid)
% analyticDoubleIntegratorTTR: Analytic double integrator time to reach.
%
%   mttr = analyticDoubleIntegratorTTR(grid)
%  
% Computes the analytic minimum time to reach each node in the grid
%   for the double integrator under optimal unit bounded control.
%
% Analytic solution from equation (7-26), p. 514 of M. Athans & P. Falb,
%   "Optimal Control", McGraw-Hill (1966).
%
% The dynamics are
%
%        \dot x    = y
%	 \dot y    = b
%
%   where input |b| \leq 1 is trying to hit the target.
%
% Parameters:
%
%   grid         Grid structure on which data is to be computed.
%
%   mttr         Minimum time to reach function.

% Copyright 2004 Ian M. Mitchell (mitchell@cs.ubc.ca).
% This software is used, copied and distributed under the licensing 
%   agreement contained in the file LICENSE in the top directory of 
%   the distribution.
%
% Ian Mitchell, 11/26/04

switchCurve = -0.5 * grid.xs{2} .* abs(grid.xs{2});

right = (grid.xs{1} > switchCurve);
left  = (grid.xs{1} < switchCurve);
on = (grid.xs{1} == switchCurve);

t1 = (grid.xs{2} + sqrt(4 * grid.xs{1} + 2 * grid.xs{2}.^2)) .* right;
t2 = (-grid.xs{2} + sqrt(-4 * grid.xs{1} + 2 * grid.xs{2}.^2)) .* left;
t3 = abs(grid.xs{2}) .* on;
mttr = t1+t2+t3;

% mttr = ((grid.xs{2} + sqrt(4 * grid.xs{1} + 2 * grid.xs{2}.^2)) .* right ...
%         + (-grid.xs{2} + sqrt(-4 * grid.xs{1} + 2 * grid.xs{2}.^2)) .* left ...
%         + abs(grid.xs{2}) .* on);
