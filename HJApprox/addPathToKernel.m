% Script file that adds the Kernel subdirectories to the path.
%
% Call this script before working with the Matlab Level Set Toolbox
%   (or place it in your startup.m file).

% Copyright 2004 Ian M. Mitchell (mitchell@cs.ubc.ca).
% This software is used, copied and distributed under the licensing 
%   agreement contained in the file LICENSE in the top directory of 
%   the distribution.
%
% Ian Mitchell, 1/13/04

% Path to ToolboxLS kernel.
addpath(genpath('../ToolboxLS-Kernel'));

% Path to Analytic solution.
addpath(genpath('../Analytic'));
