function values = interp1NO(grid, xs, data, dim, locations, method)
% interp1NO: Non-oscillatory interpolation of function values in 1D.
%
%   values = interp1NO(grid, xs, data, dim, locations, method)
%
% Interpolates function values from a 1D grid using Essentially
% Non-Oscillatory (ENO) or Weighted Essentially Non-Oscillatory (WENO)
% schemes.  For a good survey of these schemes, see Shu, "Essentially
% Non-Oscillatory and Weighted Essentially Non-Oscillatory Schemes for
% Hyperbolic Conservation Laws" in Advanced Numerical Approximation of
% Nonlinear Hyperbolic Equations (Cockburn, Johnson, Shu & Tadmor, eds),
% Lecture Notes in Mathematics 1697, pp. 325 - 432 (Springer: 1998).  Note
% that here we are interpolating function value from point samples of the
% function at the grid nodes, which is slightly different from either of
% the cases covered in that text.  The ENO3, ENO4, WENO4 and WENO6 schemes
% correspond to the ENO2, ENO3, WENO3 and WENO5 schemes from that survey
% (which are standard in the literature).
%
% This function is primarily intended to be used as a helper for
% interpnNO(), which works in any dimension.  If this function is called
% independently, be careful with the input parameters; for example, methods
% 'nearest' and 'linear' are not supported and neither is extrapolation.
%
% Input Parameters:
%
%   grid: Grid structure (see processGrid.m for details).  A grid structure
%   is required because it provides boundary conditions for populating
%   ghost cells and the spacing of the abscissae (eg: grid.dx).  Note that
%   the grid nodes are NOT used as abscissae.
%
%   xs: Array.  The abscissae in the relevant dimension at which the data are
%   given.  Note that these abscissae need not be grid nodes, although they
%   should be equally spaced by grid.dx(dim).
%
%   data: Double array.  The function values at the abscissae, over which
%   the interpolation will be performed.
%
%   dim: Integer.  Dimension in which interpolation should proceed.
%
%   locations: Array.  The coordinate locations in the relevant dimension
%   where the interpolated values should be computed.  Note that an error
%   will occur if any of these locations are outside the range of abscissae
%   contained in xs (eg: no extrapolation is allowed).
%
%   method: String.  Specifies which interpolation scheme to use.  The
%   options are:
%
%     ENO3: ENO interpolation choosing between two third order accurate
%     parabolic interpolants.
%
%     ENO4: ENO interpolation choosing between three fourth order accurate
%     cubic interpolants.
%
%     WENO4: WENO interpolation blending two third order accurate parabolic
%     interpolants, which will be fourth order accurate in smooth regions.
%
%     WENO6: WENO interpolation blending three fourth order accurate cubic
%     interpolants, which will be sixth order accurate in smooth regions.
%
% Output Parameters:
%
%     values: Double array, the same size as the arrays in the cell entries
%     of locations.  The interpolated value of the function at each
%     location.

% Copyright 2010 Ian M. Mitchell (mitchell@cs.ubc.ca).
% This software is used, copied and distributed under the licensing 
%   agreement contained in the file LICENSE in the top directory of 
%   the distribution.
%
% Ian Mitchell, 8/10/2010
% $Date: 2010-08-14 23:50:37 -0700 (Sat, 14 Aug 2010) $
% $Id: interp1NO.m 52 2010-08-15 06:50:37Z mitchell $

  %---------------------------------------------------------------------------
  %% Compute divided difference table.
  % Determine how many levels of divided differences are needed.
  switch(lower(method))
    case { 'eno3', 'weno4' }
      level = 2;
    case { 'eno4', 'weno6' }
      level = 3;
    otherwise
      error('Unknown method: %s', method)
  end
  
  dd_table = dividedDifferenceTable(grid, data, dim, level);

  %---------------------------------------------------------------------------
  %% Determine abscissa to the left of each interpolation location.
  left_index = floor((locations - xs(1)) * (1 / grid.dx(dim)));
  
  %---------------------------------------------------------------------------
  %% Compute using appropriate scheme.
  % Note that we have already caught incorrect method arguments in the
  % previous switch statement.
  switch(lower(method))
  
    %---------------------------------------------------------------------------
    %% ENO schemes.
    case { 'eno3', 'eno4' }

      % Constant term is just the function value at the left node.
      values = data(left_index);
      
    %---------------------------------------------------------------------------
    %% WENO schemes.
    case { 'weno4', 'weno6' }
      error('WENO not yet implemented');
      
  end
end

