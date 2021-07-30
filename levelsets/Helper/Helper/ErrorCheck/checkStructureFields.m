function checkStructureFields(structure, varargin)
% checkStructureFields: check that a structure contains certain fields
%
%   checkStructureFields(structure, 'field1', 'field2', ...)
%
% Generates an error if:
%   1) Structure input is not actually a structure.
%   2) Any of the field names is not present in the structure.
%
% Parameters:
%   structure    The structure in which to check for fields.
%   'field*'     Strings specifying the field names that the structure
%                  should contain.
%

% Copyright 2004 Ian M. Mitchell (mitchell@cs.ubc.ca).
% This software is used, copied and distributed under the licensing 
%   agreement contained in the file LICENSE in the top directory of 
%   the distribution.
%
% Ian Mitchell, 2/11/04

  if(isstruct(structure))
    for i = 1 : nargin - 1
      if(~isfield(structure, varargin{i}))
        error('Missing field %s in structure %s', varargin{i}, inputname(1));
      end
    end
  else
    error('%s is not a structure', inputname(1))
  end
