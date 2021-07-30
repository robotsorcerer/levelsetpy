function product = setCrossProduct(sets, dim, product)
% setCrossProduct: Build the cross product of a bunch of sets of numbers.
%
%   product = setCrossProduct(sets, dim, product)
%
% Generates a bunch of output sets (rows) each of which contains one element
% from each input set.  The entire set of output sets includes all
% combinations of input elements.
%
% Input Parameters:
%
%   sets: Column cell vector.  Each element is an input set: a vector of
%   numbers.
%
%   dim: Current dimension which is being constructed.  Used only during
%   recursion.  User should leave out this parameter.
%
%   product: The partially constructed output sets.  Used only during
%   recursion.  Users should leave out this parameter.
%
% Output Parameters:
%
%   product: A 2D array containing the output sets as rows.
  
% Copyright 2007 Ian M. Mitchell (mitchell@cs.ubc.ca).
% This software is used, copied and distributed under the licensing 
%   agreement contained in the file LICENSE in the top directory of 
%   the distribution.
%
% Created by Ian Mitchell 2007-01-18 as part of JSC 2007 submission.
% Moved 2008-08-05 to a separate file.
% $Date: 2008-08-05 10:39:10 -0700 (Tue, 05 Aug 2008) $
% $Id: setCrossProduct.m 20 2008-08-05 17:39:10Z mitchell $

  if(nargin < 3)
    dim = 1;
    product = zeros(1,0);
  end
  
  if(dim <= length(sets))
    old_product_size = size(product, 1);
    new_set_size = length(sets{dim});

    % Copy the current product into an array big enough to hold the
    % product with the next set.
    new_product = [ repmat(product, new_set_size, 1), ...
                    zeros(old_product_size * new_set_size, 1) ];

    % Copy in the members of the next set.
    for i = 1 : new_set_size
      interval = (i-1)*old_product_size+1:i*old_product_size;
      new_product(interval, end) = sets{dim}(i);
    end

    % Move on to the next set.
    product = setCrossProduct(sets, dim+1, new_product);
  end
