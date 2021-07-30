function [ yOut, schemeDataOut ] = postTimestepMask(t, yIn, schemeDataIn)
% postTimestepMask: PostTimestep routine to mask out regions of state space.
%
%  [ yOut, schemeDataOut ] = postTimestepMask(t, yIn, schemeDataIn)
%
% This routine implements the postTimestepFunc prototype for a common
%   operation: masking the evolving implicit surface function.
%
% In this case masking is accomplished by
%
%          yOut = schemeDataIn.maskFunc(yIn, schemeDataIn.maskData);
%
% If the interior is defined as the negative portion of the implicit surface
%   function, masking can be used to keep an implicitly represented
%   set from entering some region S by choosing:
%
%          maskFunc = @max
%          maskData = implicit surface function for complement of S.
%
% Note that schemeData is not modified by this function.
%
% Parameters:
%   t              Current time.
%   yIn            Input version of the level set function, in vector form.
%   schemeDataIn   Input version of a structure (see below).
%
%   yOut           Output version of the level set function, in vector form.
%   schemeDataOut  Output version of the structure (unmodified).
%
% schemeData is a structure containing data specific to this type of 
%   term approximation.  For this function it contains the field(s)
%
%   .maskFunc      Function handle which takes two arguments and returns one.
%                    The first argument will be yIn.
%                    The return value will be yOut.
%   .maskData	   Second argument to maskFunc.
%
% schemeData may contain other fields.

% Copyright 2004 Ian M. Mitchell (mitchell@cs.ubc.ca).
% This software is used, copied and distributed under the licensing 
%   agreement contained in the file LICENSE in the top directory of 
%   the distribution.
%
% Ian Mitchell 4/14/04

  checkStructureFields(schemeDataIn, 'maskFunc', 'maskData');

  % Apply the mask.
  yOut = feval(schemeDataIn.maskFunc, yIn, schemeDataIn.maskData);

  % schemeData is unchanged.
  schemeDataOut = schemeDataIn;
