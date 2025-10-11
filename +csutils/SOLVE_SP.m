function [x_hat, error] = SOLVE_SP(A, y, S, tol)
% Subspace Pursuit (SP) algorithm to recover a sparse signal.
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
   
    [~, N] = size(A); % Get length of signal x
    [~, C] = size(y); % Get number of signals

    r = y; % Initialize the residual
    T = double.empty(0,S); % Initialize the support set
    x_hat_T = zeros(S,C);  % Initialize the sparse signal estimate

    for i = 1:nchoosek(N,S) % Theoretical max you should need

        % Add top S correlated column indice to support set
        [~, J] = maxk(abs(A' * r), S);
        T_i = union(T, J');
        
        x_hat_T_i = SIGMA_S(pinv(A(:,T_i)) * y, S);
        T_i(~find(x_hat_T_i)) = [];
        x_hat_T_i(~find(x_hat_T_i)) = [];

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


function  v_s = SIGMA_S(v, s)
% SIGMA_S  zeros out all but the highest s values of v
    arguments
        v {mustBeNumeric}
        s {mustBeInteger, mustBePositive}
    end

    [~, idx] = maxk(abs(v),s);
    v_s = zeros(size(v), 'like', v);
    v_s(idx) = v(idx);
end