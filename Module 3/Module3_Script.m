clearvars; close all; clc;

%% Problem 1
function [X_sparse, X_error] = OMP(A, y, K, tol )
    arguments
        A 
        y 
        K 
        tol = 10e-6
    end
    
    [~, N] = size(A);

    x=0; r=y; I=zeros(0,K);
    for ii = 1:K
        [~, I(ii)] = max(A'*r); 
        I = sort(I);

        A_ii = A(:,I);
        x_ii = A_ii\y;
        
        r_ii =  y - A_ii*x_ii;

        norm_r =norm(r,2); norm_r_ii=norm(r_ii,2);

        if norm_r<=norm_r_ii || norm_r<=tol % Residual is getting larger or residual is small enough
            break;
        end
        
        r=r_ii; x=x_ii;
    end
    
    X_sparse = zeros(N,1);
    X_sparse(I) = x;
    X_error = norm(y - A*X_sparse, 2);
end

function [X_sparse, X_error] = SP(A, y, K, tol)
    arguments
        A 
        y 
        K 
        tol = 10e-6
    end
    
    [~, N] = size(A);

    x=zeros(N,1); r=y; s=[];
    
    while true
        if isapprox(r,0)
            break;
        end

        [~, c_ii] = maxk(A'* r, K); c_ii = sort(c_ii);
        
        s_ii = union(s, c_ii);
        
        
        x_ii = keep_max(pinv(A(:,s_ii)) * y, K);
        s_ii(x_ii==0)=[]; x_ii(x_ii==0) =[];

        r_ii =  y - A(:,s_ii)*x_ii;

        norm_r=norm(r,2); norm_r_ii=norm(r_ii,2);

        if norm_r<=norm_r_ii || norm_r<=tol % Residual is getting larger or residual is small enough
            break;
        end

        r=r_ii; x=x_ii; s=s_ii;

    end

    X_sparse = zeros(N,1);
    X_sparse(s) = x;
    X_error = norm(y - A*X_sparse, 2);
end

function zeroed_A = keep_max(v, n)
    [~, idx] = maxk(v,n);
    zeroed_A = zeros(size(v), 'like', v);
    zeroed_A(idx) = v(idx);
end

%% Problem 2
load('ps1_2022.mat');
A = {Af,Ar}; %Sensing
y = {yf,yr}; %Measurement
mat_text = {'Af', 'Ar'};
K = 3;

% A
for ii = 1:numel(A)
    [~, error] = OMP(A{ii},y{ii},K);
    fprintf("%s Error for OMP: %.4e\n", mat_text{ii}, error)
end

% B
for ii = 1:numel(A)
    [~, error] = SP(A{ii},y{ii},K);
    fprintf("%s Error for SP: %.4e\n", mat_text{ii}, error)
end



%% Problem 3
N = 256; K = 5;

tol = 10e-6;
num_measurements = 10:10:100;

for M = num_measurements
    count = 0;
    for ii = 1:100
        x=zeros(N,1); q=randsample(1:N,K); x(q)=randn(K,1);
        A=randn(M, N); A=orth(A')';

        y = A*x;
        x_sp = SP(A,y,K);
        error_ii = norm(x_sp - x,2);
        if error_ii <= tol
            count = count+1;
        end
    end
    count;
end