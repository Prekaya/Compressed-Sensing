function [x_hat, error] = SOLVE_OMP(A, y, S, tol)
% Orthogonal Matching Pursuit (OMP) algorithm to recover a sparse signal.
%
% Inputs:
%   A     - Sensing matrix (M x N)
%   y     - Measurement vector (M x C)
%   S     - Sparsity level (number of non-zero elements in x)
%   tol   - Optional tolerance to stop iterating
%
% Output:
%   x_hat - Reconstructed sparse signal (N x C)
%   error - Error between A*x_hat and y
    arguments
        A   {mustBeNumeric}
        y   {mustBeNumeric, mustHaveSameRows(A,y)}
        S   {mustBeInteger, mustBePositive}
        tol {mustBeReal, mustBePositive} = 1e-15
    end
    
    % Get dimension of sensing and measures
    [~, N] = size(A);
    [~, C] = size(y);

    r = y; % Initialize the residual
    T = double.empty(0,S); % Initialize the support set
    x_hat_T = zeros(S,C);  % Initialize the sparse signal estimate
    
    for i = 1:S

        % Add most correlated column indice to support set
        [~, J] = max(abs(A'*r));
        T_i = union(T,J);
        
        % Solve the lq to get best estimate with the support set
        x_hat_T_i = pinv(A(:,T_i))*y; 

        % Get residual for best estimate
        r_i = y - A(:,T_i)*x_hat_T_i; 
        
        if  (norm(r_i,2) >= norm(r,2)) || (norm(r, 2) < tol)
            break;     
        else
            % Update residual, support set, and sparse x then continue
            r = r_i;
            T = T_i;
            x_hat_T = x_hat_T_i;
        end
    end
    
    % Build full signal estimate vector 
    x_hat = zeros(N,C);
    if ~isempty(T)
        x_hat(T,:) = x_hat_T;
    end
    % Calculate Measurement error for full estimate
    error  = norm(y-A*x_hat, 2);
end


function mustHaveSameRows(a,b)
% Validate that a and b have the same number of rows
    if ~isequal(size(a,1), size(b,1))
        error('Inputs must be have the same number of rows.');
    end
end