function data = shapeUnion(shape1, shape2)
% shapeUnion: implicit surface function for the union of two shapes.
%
%   data = shapeUnion(shape1, shape2)
%
% Creates an implicit surface function for the union of two shapes
%   which are themselves defined by implicit surface functions.
%
% The union is created by taking the pointwise minimum of the functions.
%
% If the two shapes are defined by signed distance functions,
%   the resulting union function will be close to but not exactly
%   a signed distance function.
%
% parameters:
%   shape1      Implicit surface function data array for one shape.
%   shape2      Implicit surface function data array for the other shape.
%
%   data	Output data array (same size as shape1 and shape2) 
%                 containing an implicit surface function of the union.

% Copyright 2004 Ian M. Mitchell (mitchell@cs.ubc.ca).
% This software is used, copied and distributed under the licensing 
%   agreement contained in the file LICENSE in the top directory of 
%   the distribution.
%
% Ian Mitchell, 6/23/04

%---------------------------------------------------------------------------
data = min(shape1, shape2);

%---------------------------------------------------------------------------
% Warn the user if there is no sign change on the grid
%  (ie there will be no implicit surface to visualize).
if(all(data(:) < 0) || (all(data(:) > 0)))
  warning([ 'Implicit surface not visible because function has ' ...
            'single sign on grid' ]);
end
