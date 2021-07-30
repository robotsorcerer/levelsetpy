function data = shapeDifference(shape1, shape2)
% shapeDifference: implicit surface function for the difference of two shapes.
%
%   data = shapeDifference(shape1, shape2)
%
% Creates an implicit surface function for the difference of two shapes
%   which are themselves defined by implicit surface functions.
%
%       output = shape1 - shape2 (geometric set subtraction)
%
% The difference is created by intersecting shape1 with the complement of 
%   shape2, using pointwise maximum and negation respectively.
%
% If the two shapes are defined by signed distance functions,
%   the resulting difference function will be close to but not exactly
%   a signed distance function.
%
% parameters:
%   shape1      Implicit surface function data array for the first shape.
%   shape2      Implicit surface function data array for the subtracted shape.
%
%   data	Output data array (same size as shape1 and shape2) 
%                 containing an implicit surface function of the difference.

% Copyright 2004 Ian M. Mitchell (mitchell@cs.ubc.ca).
% This software is used, copied and distributed under the licensing 
%   agreement contained in the file LICENSE in the top directory of 
%   the distribution.
%
% Ian Mitchell, 6/23/04

%---------------------------------------------------------------------------
data = max(shape1, -shape2);

%---------------------------------------------------------------------------
% Warn the user if there is no sign change on the grid
%  (ie there will be no implicit surface to visualize).
if(all(data(:) < 0) || (all(data(:) > 0)))
  warning([ 'Implicit surface not visible because function has ' ...
            'single sign on grid' ]);
end
