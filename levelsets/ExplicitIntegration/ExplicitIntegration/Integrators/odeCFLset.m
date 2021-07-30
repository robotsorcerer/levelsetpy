function options = odeCFLset(varargin)
% odeCFLset: Create/alter options for CFL constrained ode integrators.
%
%   options = odeCFLset('name1', value1, 'name2', value2, ...)
%   options = odeCFLset(oldopts, 'name1', value1, ...)
%
% Creates a new options structure (or alters an old one) for CFL
%   constrained ODE integrators.  Basically the same as Matlab's odeset
%   but with not nearly as many options.
%
% If called with no input or output parameters, then all options,
%   their valid values and defaults are listed.
%
% Available options (options names are case insensitive):
%
%   FactorCFL     Scalar by which to multiply CFL timestep bound in order
%                   to determine the timestep to actually take.
%                   Typically in range (0,1), default = 0.5
%                   choose 0.9 for aggressive integration.
%
%   MaxStep       Maximum step size (independent of CFL).
%                   Default is REALMAX.
%
%   PostTimestep  Function handle to a routine with prototype
%                        [ yOut, schemeDataOut ] = f(t, yIn, schemeDataIn)
%                   which is called after every timestep and can be used
%                   to modify the state vector y or to modify or record
%                   information in the schemeData structure.
%                 May also be a cell vector of such function handles, in
%                   which case all function handles are called in order
%                   after each timestep.
%                 Defaults to [], which calls no function.
%
%   SingleStep    Specifies whether to exit integrator after a single
%                   CFL constrained timestep (for debugging).
%                   Either 'on' or 'off', default = 'off'.
%
%   Stats         Specifies whether to display statistics.
%                   Either 'on' or 'off', default = 'off'.
%
%   TerminalEvent Function handle to a routine with prototype
%                        [ value, schemeDataOut ] = ...
%                                            f(t, y, tOld, yOld, schemeDataIn)
%                   which is called after every timestep and can be used to
%                   halt time integration before the final time is reached.
%                   The input parameters include the state and time from
%                   the previous timestep.  If any element of the
%                   return parameter value changes sign from one timestep
%                   to the next, then integration is terminated and
%                   control is returned to the calling function.
%                 Integration cannot be terminated in this manner until
%                   after at least two timesteps.
%                 Unlike Matlab's ODE event system, no attempt is made
%                   to accurately locate the time at which the event
%                   function passed through zero.
%                 If both are present, the terminalEvent function will be
%                   called after all postTimestep functions.
%                 Defaults to [], which calls no function.

% Copyright 2005-2008 Ian M. Mitchell (mitchell@cs.ubc.ca).
% This software is used, copied and distributed under the licensing 
%   agreement contained in the file LICENSE in the top directory of 
%   the distribution.
%
% Created by Ian Mitchell, 2/6/04
% $Date: 2010-08-09 21:31:46 -0700 (Mon, 09 Aug 2010) $
% $Id: odeCFLset.m 50 2010-08-10 04:31:46Z mitchell $

  %---------------------------------------------------------------------------
  % No output, no input means caller just wants a list of available options.
  if((nargin == 0) && (nargout == 0))
      fprintf('    factorCFL: [ positive scalar {0.5} ]\n');
      fprintf('      maxStep: [ positive scalar {REALMAX} ]\n');
      fprintf([ ' postTimestep: [ function handle | '...
                'cell vector of function handles | {[]} ]\n' ]);
      fprintf('   singleStep: [ on | {off} ]\n');
      fprintf('        stats: [ on | {off} ]\n');
      fprintf('terminalEvent: [ function handle | {[]} ]\n');
      fprintf('\n');
      return;
  end
  
  %---------------------------------------------------------------------------
  % First input argument is an old options structure
  if((nargin > 0) && isstruct(varargin{1}))
    options = varargin{1};
    startArg = 2;
  else
    % Create the default options structure.
    options.factorCFL = 0.5;
    options.maxStep = realmax;
    options.postTimestep = [];
    options.singleStep = 'off';
    options.stats = 'off';
    options.terminalEvent = [];
    startArg = 1;
  end
  
  %---------------------------------------------------------------------------
  % Loop through remaining name value pairs
  for i = startArg : 2 : nargin
    name = varargin{i};
    value = varargin{i+1};

    % Remember that the case labels are lower case.
    switch(lower(name))
     case 'factorcfl'
      if(isa(value, 'double') && (prod(size(value)) == 1) && (value > 0.0))
        options.factorCFL = value;
      else
        error('FactorCFL must be a positive scalar double value');
      end

     case 'maxstep'
      if(isa(value, 'double') && (prod(size(value)) == 1) && (value > 0.0))
        options.maxStep = value;
      else
        error('MaxStep must be a positive scalar double value');
      end

     case 'posttimestep'
      if(isa(value, 'function_handle') || isempty(value))
        options.postTimestep = value;
      elseif(isa(value, 'cell'))
        for j = 1 : length(value)
          if(~isa(value{j}, 'function_handle'))
            error([ 'Each element in a postTimestep cell vector must ' ...
                    'be a function handle.' ]);
          end
        end
        options.postTimestep = value;
      else
        error([ 'PostTimestep parameter must be a function handle or ' ...
                'a cell vector of function handles.' ]);
      end

     case 'singlestep'
      if(isa(value, 'char') && (strcmp(value, 'on') || (strcmp(value, 'off'))))
        options.singleStep = value;
      else
        error('SingleStep must be one of the strings ''on'' or ''off''');
      end

     case 'stats'
      if(isa(value, 'char') && (strcmp(value, 'on') || (strcmp(value, 'off'))))
        options.stats = value;
      else
        error('Stats must be one of the strings ''on'' or ''off''');
      end

     case 'terminalevent'
      if(isa(value, 'function_handle') || isempty(value))
        options.terminalEvent = value;
      else
        error('PostTimestep parameter must be a function handle.');
      end

     otherwise
      error('Unknown odeCFL option %s', name);

    end
  end
