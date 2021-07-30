function data = shapeComplement(shape1)
% shapeComplement: implicit surface function for the complement of a shape.
%
%   data = shapeComplement(shape1)
%
% Creates an implicit surface function for the complement of a shape
%   which is itself defined by an implicit surface function.
%
% The complement is created by taking the pointwise negation of the function.
%
% If the shape is defined by a signed distance function,
%   the resulting complement function will be a signed distance function.
%
% parameters:
%   shape1      Implicit surface function data array for the shape.
%
%   data	Output data array (same size as shape1) 
%                 containing an implicit surface function of the complement.

% Copyright 2004 Ian M. Mitchell (mitchell@cs.ubc.ca).
% This software is used, copied and distributed under the licensing 
%   agreement contained in the file LICENSE in the top directory of 
%   the distribution.
%
% Ian Mitchell, 6/23/04

%---------------------------------------------------------------------------
data = -shape1;

%---------------------------------------------------------------------------
% Warn the user if there is no sign change on the grid
%  (ie there will be no implicit surface to visualize).
if(all(data(:) < 0) || (all(data(:) > 0)))
  warning([ 'Implicit surface not visible because function has ' ...
            'single sign on grid' ]);
end
