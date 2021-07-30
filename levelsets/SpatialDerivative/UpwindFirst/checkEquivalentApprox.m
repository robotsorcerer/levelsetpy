function [ relError, absError ] = checkEquivalentApprox(approx1, approx2,bound)
% checkEquivalentApprox: Checks two derivative approximations for equivalence.
%
%   [ relError, absError ] = checkEquivalentApprox(approx1, approx2, bound)
%
% Checks two derivative approximations for equivalence.
%
% A warning is generated if either of these conditions holds:
%   1) The approximation magnitude > bound
%      and the maximum relative error > bound.
%   2) The approximation magnitude < bound
%      and the maximum absolute error > bound.
%
% Normally, the return values are ignored
%   (the whole point is the warning checks).
%
% parameters:
%   approx1     An array containing one approximation.
%   approx2     An array containing the other approximation.
%   bound       The bound above which warnings are generated.
%
%   relError    The relative error at each point in the array
%                 where the magnitude > bound (NaN otherwise).
%   absError    The absolute error at each point in the array.

% Copyright 2004 Ian M. Mitchell (mitchell@cs.ubc.ca).
% This software is used, copied and distributed under the licensing 
%   agreement contained in the file LICENSE in the top directory of 
%   the distribution.
%
% Ian Mitchell, 1/23/04

% Approximate magnitude of the solution
magnitude = 0.5 * abs(approx1 + approx2);

% Which nodes deserve relative treatment, and which absolute treatment?
useRelative = find(magnitude > bound);
useAbsolute = find(magnitude <= bound);

absError = abs(approx1 - approx2);

% Be careful not to divide by too small a number.
relError = NaN * ones(size(absError));
relError(useRelative) = absError(useRelative) ./ magnitude(useRelative);

% Check that bounds are respected.
if(max(relError(useRelative)) > bound)
  warning('%s\n\t%g exceeded relative bound %g', ...
          'Error in supposedly equivalent derivative approximations', ...
          max(relError(useRelative)), bound);
end
if(max(absError(useAbsolute)) > bound)
  warning('%s\n\t%g exceeded absolute bound %g', ...
          'Error in supposedly equivalent derivative approximations', ...
          max(absError(useAbsolute)), bound);
end
