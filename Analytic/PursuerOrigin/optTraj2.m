function [ z2, z1, z12, z01 ] = ...
     twoStage(type1, type2, sigmaE1, sigmaE2, t1, t2, z0, dt)
% [ z2, z1, z12, z01 ] = ...
%    twoStage(type1, type2, sigmaE1, sigmaE2, t1, t2, z0, dt)
%
% computes a two stage trajectory starting at z0
% z2 is end point of second stage
% z1 is end point of first stage
% z12 is second stage trajectory, with timestep dt
% z01 is first stage trajectory, with timestep dt
%
% Ian Mitchell, 4/25/01

if((nargout > 2) & (nargin > 7))
  ts1 = [ 0 : dt : t1, t1 ]';
  ts2 = [ 0 : dt : t2 - t1, t2 - t1 ]';
else
  ts1 = t1;
  ts2 = t2 - t1;
end

z01 = optTraj(type1, sigmaE1, z0, ts1);
z1 = z01(end, :)';

z12 = optTraj(type2, sigmaE2, z1, ts2);
z2 = z12(end, :)';

