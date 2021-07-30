function [ varargout ] = upwindFirstENO3bHelper(grid, gdata, dim, direction)
% upwindFirstENO3bHelper: helper function for upwindFirstENO3b.
%
%  [ deriv, smooth, epsilon] = ...
%                           upwindFirstENO3bHelper(grid, gdata, dim, direction)
%
% Helper function to compute the ENO and WENO directional approximations
%   to the first derivative according to the formulae in O&F section 3.4,
%   (3.25) - (3.38).
%
% In particular, this function can compute the three ENO approximations
%   using (3.25) - (3.27) and if necessary the smoothness estimates
%   using (3.32) - (3.34) and the epsilon term (3.38).
%
% parameters:
%   grid	Grid structure (see processGrid.m for details).
%   gdata       Data array (with ghost cells added).
%   dim         Which dimension to compute derivative on.
%   direction   A scalar: -1 for left, +1 for right.
%
%   deriv       A three element cell vector containing the three
%                 ENO approximation arrays phi^i for the first derivative.
%   smooth      A three element cell vector containing the three
%                 smoothness estimate arrays S_i.
%                 (Optional, don't request it unless you need it)
%   epsilon     A single array or scalar containing the small term which
%                 guards against very small smoothness estimates.
%                 (Optional, don't request it unless you need it)

% Copyright 2004 Ian M. Mitchell (mitchell@cs.ubc.ca).
% This software is used, copied and distributed under the licensing 
%   agreement contained in the file LICENSE in the top directory of 
%   the distribution.
%
% Ian Mitchell, 1/23/03

%---------------------------------------------------------------------------
dxInv = 1 / grid.dx(dim);

% How big is the stencil?
stencil = 3;

%---------------------------------------------------------------------------
% Create cell array with array indices.
sizeData = size(gdata);
indices = cell(grid.dim, 1);
for i = 1 : grid.dim
  indices{i} = 1:sizeData(i);
end

%---------------------------------------------------------------------------
% Compute the appropriate approximations.
varargout = cell(nargout, 1);
switch(direction)
 case -1
  [ varargout{:} ] = derivativeLeft(gdata, dxInv, dim, indices, stencil);
 case +1
  [ varargout{:} ] = derivativeRight(gdata, dxInv, dim, indices, stencil);
 otherwise
  error('Invalid direction parameter %d', direction);
end


%---------------------------------------------------------------------------
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%---------------------------------------------------------------------------
function [ varargout ] = derivativeLeft(data, dxInv, dim, indices1, stencil)
% varargout = derivativeLeft(data, dxInv, dim, indices1, stencil)
%
% Helper function to compute a left directional derivative.

indices2 = indices1;

% Where does the actual data lie?
indexDer = (stencil + 1) : (size(data, dim) - stencil);

% The five v terms.
terms = 5;
v = cell(terms, 1);
for i = 1 : terms
  offset = i - 3;
  indices1{dim} = indexDer + offset;
  indices2{dim} = indexDer + offset - 1;
  v{i} = (data(indices1{:}) - data(indices2{:})) .* dxInv;
end

varargout = cell(nargout, 1);
[ varargout{:} ] = derivativeWENO(v);



%---------------------------------------------------------------------------
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%---------------------------------------------------------------------------
function [ varargout ] = derivativeRight(data, dxInv, dim, indices1, stencil)
% varargout = derivativeRight(data, dxInv, dim, indices1, stencil)
%
% helper function to compute a right directional derivative

indices2 = indices1;

% where does the actual data lie?
indexDer = (stencil + 1) : (size(data, dim) - stencil);

% the five v terms
terms = 5;
v = cell(terms, 1);
for i = 1 : terms
  offset = 3 - i;
  indices1{dim} = indexDer + offset + 1;
  indices2{dim} = indexDer + offset;
  v{i} = (data(indices1{:}) - data(indices2{:})) .* dxInv;
end

varargout = cell(nargout, 1);
[ varargout{:} ] = derivativeWENO(v);



%---------------------------------------------------------------------------
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%---------------------------------------------------------------------------
function [ varargout ] = derivativeWENO(v)
% varargout = derivativeWENO(v)
%
% Helper function to compute the WENO approximation to a derivative
%   given the five v terms.
%
% Procedure and internal parameters from Osher & Fedkiw text, pp. 33 - 37.

varargout = cell(nargout, 1);
  
%---------------------------------------------------------------------------
% First item to return is the ENO approximations.
phi1 = (1/3) * v{1} - (7/6) * v{2} + (11/6) * v{3};
phi2 = (-1/6) * v{2} + (5/6) * v{3} + (1/3) * v{4};
phi3 = (1/3) * v{3} + (5/6) * v{4} - (1/6) * v{5};
varargout{1} = { phi1; phi2; phi3 };

%---------------------------------------------------------------------------
if(nargout > 1)

  % Second item to return is the smoothness estimates.
  S1 = ((13/12) * (v{1} - 2 * v{2} + v{3}) .^2 ...
        + (1/4) * (v{1} - 4 * v{2} + 3 * v{3}) .^2);
  S2 = ((13/12) * (v{2} - 2 * v{3} + v{4}) .^2 + (1/4) * (v{2} - v{4}) .^2);
  S3 = ((13/12) * (v{3} - 2 * v{4} + v{5}) .^2 ...
        + (1/4) * (3 * v{3} - 4 * v{4} + v{5}) .^ 2);
  varargout{2} = { S1; S2; S3 };

end

%---------------------------------------------------------------------------
if(nargout > 2)

  % Third item to return is epsilon.
  
  % O&F recommends the more complicated scaled version.
  % If you know that your implicit surface function is always well
  %   scaled, you could use the simpler version.
  terms = length(v);
  if(1)
    epsilon = v{1}.^2;
    for i = 2 : terms;
      epsilon = max(epsilon, v{i}.^2);
    end
    epsilon = epsilon * 1e-6 + 1e-99;
  else
    epsilon = 1e-6;
  end
  varargout{3} = epsilon;

end
