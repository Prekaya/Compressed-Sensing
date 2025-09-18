function x_hat = SOLVE_L1(A, y)
% SOLVE_L1 recovers a sparse signal using l1-minimization via linprog.

    [~, N] = size(A);
    [~, C] = size(y);
    
    % Formulate for linprog: min f'*z s.t. Aeq*z = beq, z >= 0
    % Let x = u - v, where u,v >= 0. Then ||x||_1 = sum(u+v).
    % z = [u; v]
    
    f = ones(2*N, C); % Objective function to minimize sum(u+v)
    Aeq = [A, -A];    % Constraint A(u-v) = y
    beq = y;
    lb = zeros(2*N, C); % Lower bound u,v >= 0
    
    % Suppress verbose output
    options = optimoptions('linprog', 'Display', 'none');
    
    % Solve the linear program
    z = linprog(f, [], [], Aeq, beq, lb, [], options);
    
    if isempty(z)
        x_hat = zeros(N, 1);
        return;
    end
    
    % Reconstruct x from u and v
    x_hat = z(1:N) - z(N+1:2*N);
end