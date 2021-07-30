function data = shapeZalesakDisk(grid, center, radius, slot_width, slot_length)
% shapeZalesakDisk: create a slotted disk/sphere
%
%   data = shapeZalesakDisk(grid, center, radius, slot_width, slot_length)
%
% Creates an implicit surface function for the slotted disk from
%
%    Zalesak, "Fully Multidimensional Flux-Corrected Transport for
%    Fluids," J. Computational Physics, v.31, p. 335 (1979)
%
% This initial condition is often combined with a purely rotational velocity
% field to demonstrate the accuracy (or lack thereof) for level set methods
% on volume conservation.
%
% The slotted disk is constructed from unions and intersections of signed
% distance functions, but it is not itself a signed distance function.
%
% Note that the slot width is always along dimension 1 and the length along
% the final dimension.  The slot always opens in the negative direction.
%
% Input Parameters:
%
%   grid: Grid structure (see processGrid.m for details).
%
%   center: Vector of length grid.dim.  Specifies the center of the
%   circle/sphere from which the slot will be cut.
%
%   radius: Scalar.  Specifies the radius of the circle/sphere from which
%   the slot will be cut.
%
%   slot_width: Scalar.  Specifies the width of the slot.  Optional.
%   Default is 1/3 the radius of the circle.
%
%   slot_length: Scalar.  Specifies the length of the slot measured from the
%   bottom edge of the circle.  Optional.  Default is 5/3 the radius of the
%   circle (so almost but not quite all the way across the diameter).
%
% Output Parameters:
%
%   data: Output data array (of size grid.size) containing the implicit
%   surface function for the hyperplane.

% Copyright 2007 Ian M. Mitchell (mitchell@cs.ubc.ca).
% This software is used, copied and distributed under the licensing 
%   agreement contained in the file LICENSE in the top directory of 
%   the distribution.
%
% Ian Mitchell, 6/04/07

  %---------------------------------------------------------------------------
  if(nargin < 4)
    slot_width = radius / 3;
  end
  
  if(nargin < 5)
    slot_length = 5 * radius / 3;
  end
  

  %---------------------------------------------------------------------------
  % Start with the circle/sphere.
  circle_data = shapeSphere(grid, center, radius);
  
  %---------------------------------------------------------------------------
  % Next we need a rectangle representing the slot.
  
  % Start at the center, then offset down to the bottom of the
  % circle/sphere, and then back up half the length of the slot.
  center_offset = [ zeros(grid.dim - 1, 1); -radius + 0.5 * slot_length ];
  slot_center = center + center_offset;
  
  % The rectangle's widths are:
  %
  % Dimension 1: slot_width.
  % Dimension 2: slot_length.
  % All other dimensions: 2 * radius.
  slot_widths = [ slot_width; 2 * radius * ones(grid.dim-2, 1); slot_length ];
  
  % The rectangle representing the slot.
  slot_data = shapeRectangleByCenter(grid, slot_center, slot_widths);
  
  %---------------------------------------------------------------------------
  % To build Zalesak's disk, subtract the slot rectangle from the circle.
  data = shapeDifference(circle_data, slot_data);
