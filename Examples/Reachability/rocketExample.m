function rocketExample()
    clc; clear all;
    u_bound = 1;
    w_bound = 1;
	grid_bound = 64;
	grid_min = [-grid_bound, -grid_bound, -pi/2]';
	grid_max = [grid_bound, grid_bound, pi/2]';
	pdDims = 3;
	resolution = 100;
	N = [resolution, resolution, resolution]';
	grid = createGrid(grid_min, grid_max, N, pdDims);

    g=32; u = u_bound; w=w_bound; a=1;
    % initial value function 
    [axis_align, center, radius] = deal(2, zeros(3, 1), 1.5);
	value_init = shapeCylinder(grid, axis_align, center, radius);

    innerFunc = @termLaxFriedrichs;
    innerData.hamFunc = @hamiltonian;
    innerData.partialFunc = @dissipation;
    innerData.grid = grid;
    innerData.a = a;
    innerData.g = g;
    innerData.u_p = -1;
    innerData.u_e = 1;
    innerData.dissFunc = @artificialDissipationGLF;
    innerData.derivFunc = @upwindFirstENO2;
    integratorFunc = @odeCFL2;   
    integratorOptions = odeCFLset('factorCFL', 0.95, 'stats', 'on', 'singleStep', 'off');

    schemeFunc = @termRestrictUpdate;
    schemeData.innerFunc = innerFunc;
    schemeData.innerData = innerData;
    schemeData.positive = 0;

    t_range = [0, 1.0];
    t_now = t_range(1);
    start_time = cputime;
    
    small = 100 * eps;
    value_rolling = value_init;
    cpu_time_buffer = [];
    t_steps = (t_range(end) - t_range(1)) / 10;

    while (t_range(end) - t_now) > small * t_range(end)
        cpu_start = cputime;
        y0 = value_rolling(:);

        t_span = [t_now, min(t_range(end), t_now + t_steps)];
        [t,y] = odeCFL2(schemeFunc, t_span, y0, integratorOptions, schemeData);

        t_now = t;

        value_rolling = reshape(y, grid.shape);

        cpu_end = cputime;

        cpu_time_buffer = [cpu_time_buffer, cpu_end-cpu_start];
    end
    end_time = cputime;

    fprintf("Avg. local time: %.4f secs.\n", sum(cpu_time_buffer)/length(cpu_time_buffer))
    fprintf("Total time: %.4f secs. \n", end_time - start_time)
end

function Hxp = hamiltonian (t, value, value_derivs, finite_diff_bundle)
       %{
          H = -a p_1 \cos θ - p_2(g - a -asin θ) - \bar{u} | p_1 x + p_3 | + 
                                underline{u} | p_2 x + p_3|

          Parameters
          ==========
           value: Value function at this time step, t
           value_derivs: Spatial derivatives (finite difference) of
                        value function's grid points computed with
                        upwinding.
           finite_diff_bundle: Bundle for finite difference function
            .innerData: Bundle with the following fields:
            .partialFunc: RHS of the o.d.e of the system under consideration
                        (see function dynamics below for its impl).
            .hamFunc: Hamiltonian (this function).
            .dissFunc: artificial dissipation function.
            .derivFunc: Upwinding scheme (upwindFirstENO2).
            .innerFunc: terminal Lax Friedrichs integration scheme.
       %}
            [p1, p2, p3] = deal(value_derivs{1}, value_derivs{2}, value_derivs{3});
            
            grid =  finite_diff_bundle.grid; 
            a = finite_diff_bundle.a; 
            g = finite_diff_bundle.g; 
            u_p = finite_diff_bundle.u_p;

            p1_coeff = -a*cos(grid.xs{3}) ;
            p2_coeff = g - a - a.*sin(grid.xs{3});
            p31_coeff = abs(p1.*grid.xs{1} + p3);
            p32_coeff = abs(p2.*grid.xs{1} + p3);

            Hxp = p1.*p1_coeff + p2.*p2_coeff - u_p.*p31_coeff + u_p.*p32_coeff;
end

function alpha = dissipation(t, data, derivMin, derivMax, finite_diff_bundle, dim)
%{
   Parameters
  ==========
    dim: The dissipation of the Hamiltonian on
    he grid (see 5.11-5.12 of O&F).

      t, data, derivMin, derivMax, schemeData: other parameters
       here are merely decorators to  conform to the boilerplate
       we use in the levelsetpy toolbox.
%}
a = 1;
grid =  finite_diff_bundle.grid;  
a = finite_diff_bundle.a;
g = finite_diff_bundle.g; 
u_p = finite_diff_bundle.u_p; 
u_e = finite_diff_bundle.u_e; 

 switch dim
    case 1
         alpha = abs(-a*cos(grid.xs{3})) + abs(u_e*grid.xs{1});
    case 2
         alpha = abs(g - a -a*sin(grid.xs{3})) + abs(u_p*grid.xs{1}); 
    case 3
         alpha = abs(u_p + u_e);
 end
end
