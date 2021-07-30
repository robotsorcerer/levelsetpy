function h = visualizeLevelSet(g, data, display_type, level, title_string)
% visualizeLevelSet: Display the level set at a particular time.
%
%   h = visualizeLevelSet(g, data, display_type, level, title_string)
%
% Displays a variety of level set visualizations in dimensions 1 to 3.
% The current figure and axis is used.
%
% A warning will be generated if the requested level set is missing. For
% those display types that do not plot a level set (such as a surf plot in
% 2D), the warning can be disabled by setting parameter level to be the
% empty vector [].
%
% Note that this routine is designed to make quick, convenient plots while
% creating, running and debugging code for the Toolbox.  It is likely that
% once a routine is running properly, problem-specific visualization code
% should be added rather than depending on the rather arbitrary choices
% of visualization parameters made here, although the code in this routine
% can certainly be used as a starting point.
%
% Input parameters:
%
%   g: Grid structure.
%
%   data: Array storing the implicit surface function.  Must be the same
%   dimension as the grid.
%
%   display_type: String specifying the type of visualization (see below).
%
%   level: Double.  Which isosurface to display.  Defaults to 0.
%
%   title_string: Optional string to place in the figure title.
%
% Output parameters:
%
%   h: Handle to the graphics object created.
%
% Display type options depend on dimension.
%
% Dimension 1:
%
%   'plot': Plot the function value vs the state as a line.  If the number
%   of grid nodes is small, mark the node value as well.
%
% Dimension 2:
%
%   'contour': Show an isocontour of the function as a solid curve in two
%   dimensions.  Permits vector values for level parameter.
%
%   'surf': Plot the function value vs the two dimensional state as a
%   surface plot.
%
% Dimension 3:
%
%   'surface': Show an isosurface of the function as a solid surface in
%   three dimensions.
%
%   'slice': On slices through the x,y,z midpoints of the grid show the
%   function value through color coding.
%
%   'contourslice': On slices through the x,y,z midpoints of the grid show
%   an isocontour of the function as a solid curve.
%
%   'wireframe': Show an isosurface as a wireframe.  Essentially the same
%   as surface, but with the faces turned off and the edges turned on.

% Copyright 2007 Ian M. Mitchell (mitchell@cs.ubc.ca).
% This software is used, copied and distributed under the licensing 
%   agreement contained in the file LICENSE in the top directory of 
%   the distribution.
%
% Ian Mitchell, 6/29/04
% Modified to make use of gridnd2mesh, Ian Mitchell 5/17/07
% Subversion tags for version control purposes.
% $Date: 2011-05-16 16:06:25 -0700 (Mon, 16 May 2011) $
% $Id: visualizeLevelSet.m 66 2011-05-16 23:06:25Z mitchell $

%---------------------------------------------------------------------------
  if(nargin < 4)
    level = 0;
  end

  if((strcmp(display_type, 'contour') || strcmp(display_type, 'contourslice'))...
     && (numel(level) == 1))
    % Scalar input to contour plot should be repeated.
    level = [ level level ];
  end

  if(~isempty(level))
    if((all(data(:) < min(level(:)))) || (all(data(:) > max(level(:)))))
      warning('No implicitly defined surface exists'); %#ok<WNTAG>
    end
  end
  
%---------------------------------------------------------------------------
  switch(g.dim)
 
   %------------------------------------------------------------------------
   case 1
    switch(display_type)
     case 'plot'
      if(g.N < 20)
        % For very coarse grids, we can identify the individual nodes.
        h = plot(g.xs{1}, data, 'b-+');
      else
        h = plot(g.xs{1}, data, 'b-');
      end
     otherwise
      error('Unknown display type %s for %d dimensional system', ...
            display_type, g.dim);
    end
    
   %------------------------------------------------------------------------
   case 2
    % In 2D, the visualization routines seem to be happy to use ndgrid.
    switch(display_type)
     case 'contour'
      [ ~, h ] = contour(g.xs{1}, g.xs{2}, data, level, 'b');
      axis square;  axis manual;
     case 'surf'
      h = surf(g.xs{1}, g.xs{2}, data);
     otherwise
      error('Unknown display type %s for %d dimensional system', ...
            display_type, g.dim);    
    end
    
   %------------------------------------------------------------------------
   case 3
    % Stupid Matlab's stupid meshgrid vs ndgrid incompatibility really shows up
    % in 3D -- many of the 3D visualization routines only work for meshgrid
    % produced grids.  Therefore, we need to massage the grid and data to
    % make it work.
    [ mesh_xs, mesh_data ] = gridnd2mesh(g, data);
    
    switch(display_type)      
     case 'surface'
      h = patch(isosurface(mesh_xs{:}, mesh_data, level));
      isonormals(mesh_xs{:}, mesh_data, h);

      % This next line works without the conversion to meshgrid
      % coordinates; it appears to be the only 3D visualization code that
      % works without this conversion.
      %h = patch(isosurface(g.xs{1}, g.xs{2}, g.xs{3}, data, level));

      set(h, 'FaceColor', 'red', 'EdgeColor', 'none');

      % It would be nice to turn on a light here -- without it, the
      % surface always looks flat.  However, repeated calls to camlight
      % in the same axis produce repeated lights, which will quickly wash
      % out the pretty specular effects.  I can't find any easy way to
      % determine whether a light already exists for an axis.
      %camlight left;  camlight right;

      lighting phong;
      view(3)
     
     case 'slice'
      % For lack of a better idea of where to put the slices, we'll just slice
      % through the middle of the grid.  For final visualizations, users
      % should write their own code with a better choice of slice planes.
      avgx = mean(g.vs{1});
      avgy = mean(g.vs{2});
      avgz = mean(g.vs{3});
      h = slice(mesh_xs{:}, mesh_data, avgx, avgy, avgz);
      
     case 'contourslice'
      avgx = mean(g.vs{1});
      avgy = mean(g.vs{2});
      avgz = mean(g.vs{3});
      h = contourslice(mesh_xs{:}, mesh_data, [ avgx ], [ avgy ], [ avgz ], level); %#ok<NBRAK>
      %set(h, 'EdgeColor', 'black');
     
     case 'wireframe'
      % Because isosurface works on ndgrid, we don't need to use the converted
      % grid and data.
      h = patch(isosurface(g.xs{:}, data, level));
      set(h, 'FaceColor', 'none', 'EdgeColor', 'black');
      view(3)
 
     otherwise
      error('Unknown display type %s for %d dimensional system', ...
            display_type, g.dim);
    end
    
   %------------------------------------------------------------------------
   otherwise
    warning('Unable to display data in dimension %d', g.dim); %#ok<WNTAG>
    
  end
  
%---------------------------------------------------------------------------
  if(nargin >= 5)
    title(title_string);
  end
  
  grid on;
  drawnow;
