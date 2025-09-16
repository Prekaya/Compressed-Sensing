function [x_hat, error] = ES(A, y, S)
% Exhaustive Search algorithm for sparse signal recovery
%
% Inputs:
%   A     - Sensing matrix (M x N)
%   y     - Measurement vector (M x C)
%   S     - Sparsity level (number of non-zero elements in x)
%
% Output:
%   x_hat - Reconstructed sparse signal (N x C)
 
    arguments
        A {mustBeNumeric}
        y {mustBeNumeric, mustHaveSameRows(A,y)}
        S {mustBeInteger, mustBePositive}
    end

    % Get dimension of sensing and measures
    [~, N] = size(A);
    [~, C] = size(y);

    x_hat = zeros(N,C); % Initialize the signal estimate
    r = y; % Initialize the residual
    
    for ss = 1:S
        K = nchoosek(1:N, ss);
        for i =  1:size(K, 1)
            % Select column subset
            T = K(i,:);

            % Solve the lq to get best estimate with the support set
            x_T = pinv(A(:,T))*y; 
    
            % Get residual for best estimate
            r_T = y - A(:,T)*x_T; 
           
            % If residual didn't decrease OR we were below tolerance
            if (norm(r,2) <= norm(r_T,2)) 
                % skip update
                continue;   
            else 
                % Update residual, support set, and sparse x then continue
                r = r_T;
                x_hat(:) = 0;
                x_hat(T,:) = x_T;
            end
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