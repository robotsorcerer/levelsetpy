function h = addSlopes(point, width, styles, slopes, labels)
% addSlopes: Adds a collection of lines to demonstrate slopes on a graph.
%
%   h = addSlopes(point, width, styles, slopes, labels)
%
% When plotting the convergence properties of algorithms, it is often useful
%   to show lines corresponding to certain rates of convergence (slopes)
%   for comparison purposes.
%
% Parameters:
%
%   point        The point where the slope lines should start (ie left side).
%   width        How many units the lines should extend in the x-direction.
%   styles       A line style (or cell vector of line styles).
%                  Defaults to 'b:'.
%   slopes       A vector specifying which slope line(s) to add.
%                  Defaults to [ 0, -1 ].
%   labels       A cell vector of labels to place to the right of the 
%                  slope lines.  Defaults to { ''; 'first order' }.
%
%   h            An array of handles.
%                  First column contains the line handles for the slopes.
%                  Second column contains the text handles for the labels.

% Copyright 2004 Ian M. Mitchell (mitchell@cs.ubc.ca).
% This software is used, copied and distributed under the licensing 
%   agreement contained in the file LICENSE in the top directory of 
%   the distribution.
%
% Ian Mitchell, 12/15/03

if(nargin < 3)
  styles = 'b:';
end

if(nargin < 4)
  slopes = [ 0; -1 ];
  labels = { ''; 'first order' };
elseif(nargin == 4)
  labels = cell(length(slopes), 1);
end

lines = length(slopes);

if(~iscell(styles))
  stylesCell = cell(lines, 1);
  for i = 1 : lines
    stylesCell{i} = styles;
  end
end

h = zeros(lines, 2);

for i = 1 : lines

  xpos = [ point(1), point(1) + width ];
  if(strcmp(get(gca, 'YScale'), 'log'))
    ratio = (point(1) + width) / point(1);
    ypos = [ point(2), point(2) * ratio^slopes(i) ];
  else
    ypos = [ point(2), point(2) + slopes(i) * width ];
  end

  h(i, 1) = plot(xpos, ypos, stylesCell{i});
  h(i, 2) = text(xpos(2) + 0.1 * width, ypos(2), labels{i});

end
