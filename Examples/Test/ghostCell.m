% Script file to test the various routines for adding ghost cells 
%   in dimensions one and two.
%
% Uses a portion of a sin function for data.

% Copyright 2004 Ian M. Mitchell (mitchell@cs.ubc.ca).
% This software is used, copied and distributed under the licensing 
%   agreement contained in the file LICENSE in the top directory of 
%   the distribution.
%
% Ian Mitchell, 1/13/04

%run('../addPathToKernel');

% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
% Which one of the routines would you like to test? (comment out the rest)
% which = 'periodic';
% which = 'extrapolate';
% which = 'dirichlet';
% which = 'neumann';

% In how many dimensions?
dims = 2;

% Test which dimension?
testDim = 2;

% How many ghost cells?
width = 2;

% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
% Create a grid.
clear g;
g.dim = dims;
g.min = zeros(g.dim, 1);
g.max = ones(g.dim, 1);
g.dx = 1 / 20;

% details depend on the type of ghost cell
switch(which)
  case 'periodic'
    g.max = g.max - g.dx;
    g.bdry = @addGhostPeriodic;
    % no need for bdryData

  case 'extrapolate'
    g.bdry = @addGhostExtrapolate;
    g.bdryData.towardZero = 0;

  case 'dirichlet'
    g.bdry = @addGhostDirichlet;
    g.bdryData.lowerValue = 0;
    g.bdryData.upperValue = 1;

  case 'neumann'
    g.bdry = @addGhostNeumann;
    g.bdryData.lowerSlope = 0 * g.dx;
    g.bdryData.upperSlope = 1 * g.dx;

  otherwise
    error('Unknown type of ghost cells %s', which);

end

g = processGrid(g);

% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
% Create some data.
switch(dims)
  case 1
    dataIn = sin(pi * (g.xs{1} + 0.25));

  case 2
    dataIn = sin(pi * (g.xs{1} + 0.25)) + g.xs{2};

  otherwise
    error('No data function specified for dimension %d', dims);
end


% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
% Check that the ghost cells are added properly
dataOut = feval(g.bdry{testDim}, dataIn, testDim, width, g.bdryData{testDim});

switch(dims)
  case 1
    plot(dataOut, 'b*');

  case 2
    surf(dataOut);

  otherwise
    error('No data visualization specified for dimension %d', dims);
end
