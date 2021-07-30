function [ line_handles, node_handles ] = ...
                        visualizeGrid(grid, node_style, line_style, dims)
% visualizeGrid: show the grid lines and nodes in a plot.
%
%    [ line_handles, node_handles ] = ...
%                       visualizeGrid(grid, node_style, line_style, dims)
%
% Plots lines and optionally marks nodes on the current plot for the
% specified grid.  Useful to visualizing the grid against various aspects of
% the level set; for example, to see whether there might be grid related
% artifacts or for qualitative inspection of subgrid resoltuion.
%
% Input Parameters:
%
%   grid: Standard toolbox grid structure.
%
%   node_style: One or two character string specifying how nodes should be
%   displayed.  Use the color and symbol codes from Matlab's plot command.
%   Optional, default is no node display.  Use [] to indicate no node
%   display if you want to enter parameters after this one.
%
%   line_style: One or two character string specifying how the grid lines
%   should be displayed, or a three element vector specifying a color (in
%   RGB coordinates).  For the character string use the color and line style
%   codes from Matlab's plot command.  Optional, default is solid lines in a
%   light grey color (so that they can be distinguished from the result of
%   Matlab's grid('on') command).  If a color vector is specified, a solid
%   line in that color will be used.
%
%   dims: Vector.  For two and three dimensional grids the plotting is
%   straightforward.  For grids in higher dimensions this parameter is used
%   to choose which projection of the grid will be visualized (it can also
%   be used to visualize a 2D projection of a 3D grid).  Optional, default
%   is to visualize all the dimensions of a 2D or 3D grid, and generate an
%   error for higher dimensional grids.
%
% Output Parameters:
%
%   line_handles: A vector of graphics handles for the lines in the plot.
%
%   node_handles: A vector of graphics handles for the node symbols in the
%   plot.  If the nodes were not plotted, this parameter will be [].

% Created by Ian Mitchell, 2008-05-23.
% $Date: 2008-05-24 17:50:39 -0700 (Sat, 24 May 2008) $
% $Id: visualizeGrid.m 10 2008-05-25 00:50:39Z mitchell $

  % - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  if(nargin < 2)
    node_style = [];
  end
  if(nargin < 3)
    % We actually distinguish below between the style and the color of
    % the grid lines.  Therefore, unpack the user's input if necessary.
    line_style = '-';
    line_color = 0.9 * ones(1,3);
  elseif ischar(line_style)
    % Line style is already set.
    line_color = [];
  else
    % User actually provided the line color.
    line_color = line_style;
    line_style = '-';
  end
  if(nargin < 4)
    if(grid.dim <= 3)
      dims = 1 : grid.dim;
    else
      error([ 'Projection via the dimensions parameter required for ' ...
              'grids with dimension four or more.' ]);
    end
  end

  % - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  % Sample the current setting of hold, so that we can put it back at the
  % end.
  holding = ishold;
  % Now turn on the hold so that we don't delete our plot objects.
  hold on;
  
  % - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  switch length(dims)
    
   % - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   case 1
    % I'm not sure why this code would be executed, so I'm not sure what a good
    % visualization might be.  For now, I'll generate an error.  If a
    % situation arises where this visualization is necessary, then I'll
    % implement it appropriately.
    error('Visualization for 1D grid or projection is not yet implemented.');

   % - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   case 2
    line_handles1 = plot([ grid.min(dims(1)) * ones(1, grid.N(dims(2))); ...
                           grid.max(dims(1)) * ones(1, grid.N(dims(2))) ], ...
                         repmat(grid.vs{dims(2)}', 2, 1), line_style);
    line_handles2 = plot(repmat(grid.vs{dims(1)}', 2, 1), ...
                         [ grid.min(dims(2)) * ones(1, grid.N(dims(1))); ...
                           grid.max(dims(2)) * ones(1, grid.N(dims(1))) ], ...
                         line_style);
    line_handles = [ line_handles1; line_handles2 ];
    % Adjust the color if necessary
    if ~isempty(line_color)
      set(line_handles, 'Color', line_color);
    end
    
    % Plotting the nodes is trivial.
    if isempty(node_style)
      node_handles = [];
    else
      node_handles = plot(grid.xs{dims(1)}(:), ...
                          grid.xs{dims(2)}(:), node_style);
    end
    
   % - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   case 3
    % I think that the best visualization in this case is similar to Matlab's:
    % show the grid lines against the back walls of an imaginary cube
    % containing the grid.  Of course I still need to implement it...
    error('Visualization for 3D grid or projection is not yet implemented.');
    
   % - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   otherwise
    error('No visualization for projections of more than three dimensions.');
  end

  % - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  % Return the hold state to its former value if necessary.
  if ~holding
    hold off;
  end
