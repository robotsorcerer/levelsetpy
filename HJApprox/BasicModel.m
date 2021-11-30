classdef BasicModel
  % BasicModel: Class for game of two identical vehicles dynamics
  %
  % This class implements a model of the game of two identical vehicles
  % described in [Merz 1972] and [Mitchell 2001].  The two vehicles are
  % Dubins car models.  The game is that the pursuer vehicle seeks to cause
  % a collision while the evader seeks to avoid it.  The state space is the
  % relative pose, with the evader fixed at the origin.
  %
  % This code is intended to be used to compute the robust controlled
  % backward reach set (RCBRT) using ToolboxLS.
  %
  % For more information, see [Mitchell 2020].
  
  % Created by Ian Mitchell, 2020-05-18
 
  %----------------------------------------------------------------------
  %----------------------------------------------------------------------
  properties
    
    % - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    %% Model parameters.

    % Bounds on the inputs.  For identical vehicles these values should be
    % the same for pursuer and evader.  For now we will restrict to Dubins
    % car vehicles, so the linear velocities are fixed.
    v_e
    v_p
    omega_e_bound
    omega_p_bound

    % Time interval over which to compute the RCBRT.
    time_interval
    
    % Radius of the target set.
    radius
    
    % States over which this robot object will be evaluated.
    grid
    
    % Some stuff that will be convenient to precompute.  In this case,
    % Hamiltonian terms that do not depend on inputs, and bounds on the
    % partials.
    p1_term
    p2_term
    alpha
    
  end % Properties.
  
  %----------------------------------------------------------------------
  %----------------------------------------------------------------------
  methods
    
    %----------------------------------------------------------------------
    function obj = BasicModel(omega_bounds, v_bounds, radius, resolution)
      % BasicModel: Constructor for the BasicModel class.
      %
      %   obj = BasicModel(omega_bounds, v_bounds, radius, resolution)
      %
      % Note that the grid bounds are not among the arguments.  It may be
      % necessary to adjust the grid bounds to fully capture the RCBRT if
      % values other than the default for omega_bounds, v_bounds and/or
      % radius are used.
      %
      % Likewise the time interval may need to be adjusted for other
      % problem parameters.
      %
      % Input Parameters:
      %
      %   omega_bounds: Default = +1.  One of
      %
      %     Positive scalar.  Used as bound on angular velocity of both
      %     vehicles.
      %
      %     Positive vector length 2.  First entry is bound on angular
      %     velocity of evader, second bound on angular velocity of
      %     pursuer.
      %
      %   v_bounds: Default = +1.  One of
      %
      %     Positive scalar.  Used as fixed value of linear velocity of
      %     both vehicles.
      %
      %     Positive vector length 2.  First entry is linear velocity of evader,
      %     second linear velocity of pursuer.
      %
      %   radius: Default = 0.5.  Positive scalar.  Radius of the target set.
      %
      %   resolution: Default = 100.  One of
      %
      %     Positive integer.  Used as the number of grid nodes in the x_1
      %     and x_3 dimensions.  An appropriately scaled value is used for
      %     the x_2 dimension in order to make the grid spacing in x_1 and
      %     x_2 the same.
      %
      %     Positive integer vector of length 3.  Used as the number of
      %     grid nodes in each dimension.
      %
      % Output Parameters:
      %
      %   obj: BasicModel object.  You can access the grid that is
      %   constructed through obj.getGrid().

      if(nargin < 1)
        omega_bounds = +1;
      end
      if(nargin < 2)
        v_bounds = +1;
      end
      if(nargin < 3)
        radius = 0.5;
      end
      if(nargin < 4)
        resolution = 100;
      end

      assert(all(omega_bounds(:) > 0));
      if(length(omega_bounds) == 1)
        obj.omega_e_bound = omega_bounds;
        obj.omega_p_bound = omega_bounds;
      else
        assert(length(omega_bounds) == 2);
        obj.omega_e_bound = omega_bounds(1);
        obj.omega_p_bound = omega_bounds(2);
      end

      assert(all(v_bounds(:) > 0));
      if(length(v_bounds) == 1)
        obj.v_e = v_bounds;
        obj.v_p = v_bounds;
      else
        assert(length(v_bounds) == 2);
        obj.v_e = v_bounds(1);
        obj.v_p = v_bounds(2);
      end

      assert(radius > 0);
      obj.radius = radius;

      assert(resolution > 0);
      grid.dim = 3;
      grid.min = [ -0.75; -1.25; -pi ];
      grid.max = [ +3.25; +1.25; +pi ];
      grid.bdry = { @addGhostExtrapolate; @addGhostExtrapolate; @addGhostPeriodic };
      % Roughly equal dx in x and y (so different N).
      grid.N = [ resolution; 
        ceil(resolution * (grid.max(2) - grid.min(2)) / (grid.max(1) - grid.min(1))); 
        resolution - 1 ];
      % Need to trim max bound in x_3 (since the BC are periodic in this dimension).
      grid.max(3) = grid.max(3) * (1 - 2 / grid.N(3));
      obj.grid = processGrid(grid);

      % Time interval for the RCBRT.
      obj.time_interval = [ 0, 2.5 ];
      
      % Precompute the useful data.
      obj.p1_term = obj.v_e - obj.v_p * cos(obj.grid.xs{3});
      obj.p2_term = -obj.v_p * sin(obj.grid.xs{3});
      obj.alpha = { abs(obj.p1_term) + abs(obj.omega_e_bound * obj.grid.xs{2}) ...
        abs(obj.p2_term) + abs(obj.omega_e_bound * obj.grid.xs{1}) ...
        obj.omega_e_bound + obj.omega_p_bound };
              
    end % BasicModel() constructor.

    %----------------------------------------------------------------------
    function data0 = getTarget(obj)
      % getTarget: Create level set function for the target set.
      %
      %   data0 = getTarget(obj)
      %
      % Input Parameters: None.
      %
      % Output Parameters:
      %
      %   data0: Array of size grid.shape.  Implicit surface function
      %   defining the target set for this RCBRT.
      
      data0 = shapeCylinder(obj.grid, 3, [ 0; 0; 0 ], obj.radius);
    
    end % getTarget().

    %----------------------------------------------------------------------
    function grid = getGrid(obj)
      % getGrid: Get the grid on which to compute the RCBRT.
      %
      %   grid = getTarget(obj)
      %
      % Input Parameters: None.
      %
      % Output Parameters:
      %
      %   grid: Standard grid structure.
      
      grid = obj.grid;
    
    end % getGrid().

    %----------------------------------------------------------------------
    function time_interval = getTimeInterval(obj)
      % getGrid: Get the time interval on which to compute the RCBRT.
      %
      %   time_interval = getTarget(obj)
      %
      % Input Parameters: None.
      %
      % Output Parameters:
      %
      %   time_interval: Vector of length 2 containing the initial and
      %   final time for computing the RCBRT.
      
      time_interval = obj.time_interval;
    
    end % getTimeInterval().

    %----------------------------------------------------------------------
    function [ ham_value, schemeData ] = hamFunc(obj, ~, ~, deriv, schemeData)
      % hamFunc: analytic Hamiltonian.
      %
      %   [ ham_value, schemeData ] = hamFunc(t, data, deriv, schemeData)
      %
      % This function implements the hamFunc prototype for this model.
      %
      % Input parameters:
      %
      %   t: Time at beginning of timestep.
      %
      %   data: Data array.
      %
      %   deriv: Cell vector of the costate (\grad \phi).
      %
      %   schemeData: A structure (see below).
      %
      % Output parameters:
      %
      %   ham_value: The analytic hamiltonian.
      %
      %   schemeData: The schemeData structure (not modified).
      %
      % schemeData is a structure containing data specific to this
      % Hamiltonian.  For this particular Hamiltonian, it is not needed.
      
      ham_value = deriv{1} .* obj.p1_term + deriv{2} .* obj.p2_term ...
        - obj.omega_e_bound * abs(deriv{1} .* obj.grid.xs{2} - deriv{2} .* obj.grid.xs{1} - deriv{3}) ...
        + obj.omega_p_bound * abs(deriv{3});

    end % function hamFunc().

    %----------------------------------------------------------------------
    function alpha = partialFunc(obj, ~, ~, ~, ~, ~, dim)
    % partialFunc: Hamiltonian partial function.
    %
    %   alpha = partialFunc(t, data, derivMin, derivMax, schemeData, dim)
    %
    % This function implements the partialFunc prototype for this model.
    %
    % It calculates the extrema of the absolute value of the partials of the 
    % analytic Hamiltonian with respect to the costate (gradient).
    %
    % Parameters:
    %
    %   t: Time at beginning of timestep.
    %
    %   data: Data array.
    %
    %   derivMin: Cell vector of minimum values of the costate (\grad \phi).
    %
    %   derivMax: Cell vector of maximum values of the costate (\grad \phi).
    %
    %   schemeData: A structure (see below).
    %
    %   dim: Dimension in which the partial derivatives is taken.
    %
    % Output Parameters:
    %
    %   alpha: Maximum absolute value of the partial of the Hamiltonian with
    %   respect to the costate in dimension dim for the specified range of
    %   costate values (O&F equation 5.12). Note that alpha can (and should) be
    %   evaluated separately at each node of the grid.
    %
    % schemeData is a structure containing data specific to this Hamiltonian
    % For this function it does not contain anything useful.

    assert((dim > 0) && (dim <= 3));
    alpha = obj.alpha{dim};
    
    end % function partialFunc().
    
  end % methods.
  
end % classdef.
