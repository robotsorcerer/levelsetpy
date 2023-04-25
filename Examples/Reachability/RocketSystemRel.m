classdef RocketSystemRel % (grid=[], u_bound=5, w_bound=5, a=1, g=64) 
    properties 
        grid; % = grid;
        cur_state; % =grid.xs;
        u; % = u_bound;
        w; % = w_bound;
        ae; % = a;
        ap; % = a;
        a; % = a;
        g; % = g;
        u_e; % = u_bound;
        u_p; % = -u_bound;
    end

    methods
        function [Hxp] = hamiltonian (self, t, value, value_derivs, finite_diff_bundle)
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
            arguments 
                self (1,1) RocketSystemRel
                t    (1,1) {mustBeReal, mustBeNonNeegative}
                value 
                value_derivs
                finite_diff_bundle
            end

            [p1, p2, p3] = deal(value_derivs(1), value_derivs(2), value_derivs(3));

            p1_coeff = -self.a*cos(self.grid.xs(3)) ;
            p2_coeff = self.g - self.a - self.a*sin(self.grid.xs(3));
            p31_coeff = abs(p1*self.grid.xs(1) + p3);
            p32_coeff = abs(p2*self.grid.xs(1) + p3);

            Hxp = p1*p1_coeff + p2*p2_coeff - self.u_p*p31_coeff + self.u_p*p32_coeff;
        end

        function alpha = dissipation(self, t, data, derivMin, derivMax, schemeData, dim)
            arguments 
                self (1,1) RocketSystemRel
                t    (1,1) {mustBeReal, mustBeNonNeegative}
                data
                derivMin
                derivMax
                schemeData
                dim (1,1)
            end
            %{
            Parameters
            ==========
                dim: The dissipation of the Hamiltonian on
                the grid (see 5.11-5.12 of O&F).

                t, data, derivMin, derivMax, schemeData: other parameters
                here are merely decorators to  conform to the boilerplate
                we use in the levelsetpy toolbox.
            %}
            switch dim
                case dim==0
                    alpha = abs(-self.a*cos(self.grid.xs(3))) + abs(self.u_e*self.grid.xs(1));
                case dim==1
                    alpha = abs(self.g - self.a -self.a*sin(self.grid.xs(3))) + abs(self.u_p*self.grid.xs(1)); 
                case dim==2
                    alpha = abs(self.u_p + self.u_e);
            end
        end
    end
end