function data = signedDistanceIterative(grid, data0, accuracy, tMax, errorMax)
% signedDistanceIterative: Create a signed distance function iteratively.
%
%   data = signedDistanceIterative(grid, data0, accuracy, tMax, errorMax)
%
% Converts an implicit surface function into a signed distance function
%   by iterative solution of the reinitialization equation.
%
% Iterations continue to a fixed time or until the average relative change
%   in the function value between iterations drops low enough, whichever
%   comes first.
%
% In the reinitialization equation, information flows outward from the zero
%   level set at "speed" one, so to get a signed distance function in a band
%   of at least 10 grid cells around the zero level set, choose a minimum
%   tMax = 10 * max(grid.dx).
%
% Parameters:
%
%   grid	 Grid structure.
%   data0        Implicit surface function.
%   accuracy     Controls the order of approximations.
%                  'low'         Use odeCFL1 and upwindFirstFirst.
%                  'medium'      Use odeCFL2 and upwindFirstENO2 (default).
%                  'high'        Use odeCFL3 and upwindFirstENO3.
%                  'veryHigh'    Use odeCFL3 and upwindFirstWENO5.
%   tMax         Time at which to halt the reinitialization iteration
%                  (default = max(grid.max - grid.min)).
%                  If tMax < 0, it is interpreted as the number of CFL
%                  limited reinitialization timesteps to take:
%                  number of steps = -round(tMax).
%   errorMax     If the average update of nodes drops below
%                  errorMax * max(grid.dx), then assume that 
%                  reinitialization has converged and return early 
%                  (default = 1e-3).
%
%
%   data         Signed distance function.

% Copyright 2005 Ian M. Mitchell (mitchell@cs.ubc.ca).
% This software is used, copied and distributed under the licensing 
%   agreement contained in the file LICENSE in the top directory of 
%   the distribution.
%
% Ian Mitchell, 2/14/04
% Modified to accept vector level sets, Ian Mitchell 2/16/05

%---------------------------------------------------------------------------
% How close (relative) do we need to get to tMax to be considered finished?
small = 100 * eps;

% How aggressive should we be with the CFL condition?
factorCFL = 0.95;

%---------------------------------------------------------------------------
% Set defaults.
if(nargin < 3)
  accuracy = 'medium';
end

if(nargin < 4)
  tMax = max(grid.max - grid.min);
end

if(tMax < 0)
  stepMax = -round(tMax);
  tMax = +inf; % prod(grid.N) * max(grid.max - grid.min);
else
  stepMax = +inf;
end

if(nargin < 5)
  errorMax = 1e-3;
end

%---------------------------------------------------------------------------
% If this is a vector level set, each element of the vector is
%   reinitialized independently.
if(iscell(data0))
  data = cell(length(data0), 1);
  for i = 1 : length(data0)
    if(iscell(grid))
      data{i} = signedDistanceIterative(grid{i}, data0{i}, accuracy, ...
                                        tMax, errorMax);
    else
      data{i} = signedDistanceIterative(grid, data0{i}, accuracy, ...
                                        tMax, errorMax);
    end
  end
end

%---------------------------------------------------------------------------
% Set up spatial approximation scheme.
schemeFunc = @termReinit;
schemeData.grid = grid;
% Just in case original data is in column vector format.
schemeData.initial = reshape(data0, grid.shape);

% Set up time approximation scheme.
%   Single step so that we can check convergence criterion.
integratorOptions = odeCFLset('factorCFL', factorCFL, 'singleStep', 'on');

% Choose approximations at appropriate level of accuracy.
switch(accuracy)
 case 'low'
  schemeData.derivFunc = @upwindFirstFirst;
  integratorFunc = @odeCFL1;
 case 'medium'
  schemeData.derivFunc = @upwindFirstENO2;
  integratorFunc = @odeCFL2;
 case 'high'
  schemeData.derivFunc = @upwindFirstENO3;
  integratorFunc = @odeCFL3;
 case 'veryHigh'
  schemeData.derivFunc = @upwindFirstWENO5;
  integratorFunc = @odeCFL3;
 otherwise
  error('Unknown accuracy level %s', accuracy);
end


%---------------------------------------------------------------------------
% Convergence criteria
deltaMax = errorMax * max(grid.dx) * prod(grid.N);

%---------------------------------------------------------------------------
% Reshape data array into column vector for ode solver call (if necessary).
dataSize = size(data0);
if((length(dataSize) == 2) ...
   && (dataSize(1) == prod(grid.N)) && (dataSize(2) == 1))
  % Data is already in column vector form.
  y = data0;
  reshaped = 0;
elseif((length(dataSize) == length(grid.shape)) && all(dataSize == grid.shape))
  % Reshape to column vector form.
  y = data0(:);
  reshaped = 1;
else
  error('Data array is not the same size as grid');
end

%---------------------------------------------------------------------------
% Loop until tMax (subject to a little roundoff) or stepMax.
tNow = 0;
step = 0;
while((tMax - tNow >= small * tMax) & (step < stepMax))

  % Check for convergence (except for the first loop).
  if((tNow > 0) && (norm(y - y0, 1) < deltaMax))
    break;
  end

  % Always try to finish (in fact, single timesteps are taken).
  tSpan = [ tNow, tMax ];
  
  % Take a single timestep.
  y0 = y;
  [ t y ] = feval(integratorFunc, schemeFunc, tSpan, y0,...
                  integratorOptions, schemeData);

  tNow = t(end);
  step = step + 1;
  
end


if(reshaped)
  % Reshape the column vector back into the appropriate form.
  data = reshape(y, grid.shape);
else
  data = y;
end
