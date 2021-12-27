function [ranks] = min_proj_error(X, V, eps, plot)
% Using a user defined projection error as a criteria,
% calculate the minimum projection error onto a subspace of the 
% value function X. 
%
%   Parameters:
%   -----------
%       X(matrix of doubles): Payoff functional; 
%       V (matrix of doubles): (Dominant) Left singular vectors;
%       eps (float): (User-defined) projection error;
%       plot(bool):  show the projection error on a plot?
%
%   Returns:
%   --------
%   ranks: best rank approximation of X that admits the tolerable decomp
%            error
%       Basically, it is a projection of X onto one of its subspaces.
%     if isscalar(eps)
%         eps = [eps];
%     end
    
    X_norm = norm(X, 'fro');
    rs     = 2:size(V, 2);
    errors = zeros(size(rs));
    for r=rs
        Vr = V(:,1:r);  % if it were a tensor, this decomp should be done in the for loop for each mode of the tensor
        errors(r-1) = (norm(X- Vr * Vr' * X, 'fro') / (X_norm));
    end
    
    ranks = [nnz(errors>eps)+1];
    
    if plot
        slg = semilogy(rs, errors)
        xlim([0, size(V, 2)]);
        for ep = [eps; ranks]
            yline([ep(1), 0], '-.r', {'User defined tolerance'}, 'LineWidth', 2.5);
            xline(ep(2), '--b',{'Acceptable','Order'}, 'LineWidth', 2.5);
        end
        
        slg(1).LineWidth = 3.5;
        slg(1).LineStyle = '-.';
        slg(1).Color = '#FFA500';
        
        grid('on');        
        ylim('auto');
        title("Decomposition's Projection Error", 'FontWeight', 'bold');
        xlabel("PGD reduced basis rank r", 'FontWeight', 'bold');
        ylabel("Projection error", 'FontWeight', 'bold');
        %legend('Error per Singular Vectors Mode','Location','best');
        
    end
    
        