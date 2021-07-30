function near = isNearInterface(data, interface_level, strict_opposite)
% isNearInterface: true if a node has a neighbor across the interface.
%
%   near = isNearInterface(data, interface_level, strict_opposite)
%
% For each node in the data array, determines whether that node has any
% neighbors (left/right in each dimension) which lie on the other side of
% the interface.  If the interface is the zero level set, then nodes are
% next to the interface if any of their neighbors have the opposite sign.
% Nodes lying precisely on the interface are "near the interface," but
% whether their neighbors are also near depends on the input argument
% strict_opposite.
%
% Input Parameters:
%
%   data: Array of values for each node in the grid.  Note that the grid
%   itself is not necessary for this calculation.
%
%   interface_level: Scalar.  Value which represents the "interface."
%   Optional.  Default = 0.
%
%   strict_opposite: Boolean.  Should "neighbor on the opposite side"
%   include neighbors lying precisely on the interface?  Optional.  Default
%   is 0 (a neighbor lying on the interface is "opposite" for all nodes that
%   are not on the interface).  Note that nodes which lie on the interface
%   are ALWAYS "near the interface," regardless of the value of this
%   parameters; this parameter only affects their neighbors.
%
% Output Parameters:
%
%   near: Boolean array, same size as data.  A node's value is 1 if that
%   node is on the interface or if that node has a neighbor on the
%   opposite side of the interface.

% Copyright 2007 Ian M. Mitchell (mitchell@cs.ubc.ca).
% This software is used, copied and distributed under the licensing 
%   agreement contained in the file LICENSE in the top directory of 
%   the distribution.
%
% Ian Mitchell, 5/5/07

%---------------------------------------------------------------------------
if(nargin < 2)
  interface_level = 0;
end

if(nargin < 3)
  strict_opposite = 0;
end

% All we care about is on which side of the interface a node lies.
sign_data = sign(data - interface_level);

% Nodes exactly on the interface are "near"
near = (sign_data == 0);

% To compare against neighbors, we need some index cell vectors.
data_dims = ndims(data);
data_size = size(data);
indexL = cell(data_dims, 1);
for d = 1 : data_dims
  indexL{d} = 1 : data_size(d);
end
indexR = indexL;

% Work through dimensions, looking left and right for sign differences.
% Neighbors are not on the "opposite side" if they are directly on the
% interface.
for d = 1 : data_dims

  % Offset the index arrays for this dimension.
  indexL{d} = 1 : data_size(d) - 1;
  indexR{d} = 2 : data_size(d);

  % Find the nodes near the interface.

  if strict_opposite
    % Neighbors on the interface don't count.
    near_d = (sign_data(indexL{:}).*sign_data(indexR{:}) < 0);
  else
    % Any neighbor on or across the interface counts.
    near_d = (sign_data(indexL{:}) ~= sign_data(indexR{:}));
  end

  % If we detected a nearness, it applies to both the node on the left
  % and the node on the right.
  near(indexL{:}) = near(indexL{:}) | near_d;
  near(indexR{:}) = near(indexR{:}) | near_d;

  % Reset the index arrays for the next dimension.
  indexL{d} = 1 : data_size(d);
  indexR{d} = 1 : data_size(d);

end
  
