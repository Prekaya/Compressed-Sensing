function [x_hat, error] = SP(A, y, S, tol)
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
        A {mustBeNumeric}
        y {mustBeNumeric, mustHaveSameRows(A,y)}
        S {mustBeInteger, mustBePositive}
        tol = 1e-15
    end
    
    % Get dimension of sensing and measures
    [~, N] = size(A);
    [~, C] = size(y);

    x_hat = zeros(N,C); % Initialize the signal estimate
    r = y; % Initialize the residual
    T = double.empty(0,S); % Initialize the support set
    used_A = A;
    for i = 1:nchoosek(N,S) % Theoretical max you should need
        % Add top S correlated column indice to support set
        [~, J] = maxk(abs(used_A' * r), S);
        T = union(T, J');
        
        x_T = SIGMA_S(pinv(A(:,T)) * y, S);
        T(~find(x_T)) = []; x_T(~find(x_T)) = [];

        % Get residual for best estimate
        r_T = y - A(:,T)*x_T; 

        % If residual didn't decrease OR we were below tolerance STOP
        if (norm(r,2) <= norm(r_T,2))  || (norm(A*x_hat - y, 2) < tol)
            % keep what we had and exit
            break;   
        else 
            % Update residual, support set, and sparse x then continue
            r = r_T;
            x_hat(:) = 0;
            x_hat(T,:) = x_T;
            used_A(:,T) = nan;
        end
    end

    % Error for final estimate
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