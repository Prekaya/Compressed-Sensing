clearvars; close all; clc;
load('ps1_2022.mat');

A = {Af, Ar};
y = {yf, yr};
mat_text = {'Af', 'Ar'};
S = 3;

%% Problem 1
for ii = 1:numel(A)
    [~, error] = exhaustive_search(A{ii}, y{ii}, S);
    fprintf("Error for %s: %.4e\n", mat_text{ii}, error)
end

%% Problem 2 
% Af array is mirrored Ar array seems normal
% Not really sure if something specific but the matrix isn't full ranked 
% and the linearly independent columns are not that different from each
% other
% rref(round(Af,sig_fig))

function [X_sparse, X_error] = exhaustive_search(A, y, S)
    [~, N] = size(A,1,2);

    x=0; error = Inf; x_ind=[];
    for ss = 1:S
        K = nchoosek((1:N), ss);
        for ii =  1:size(K, 1)
            I = K(ii,:);
            A_ii = A(:,I);
            x_ii = A_ii\y;

            error_ii = norm(y - A_ii*x_ii,2);
        
            if error_ii < error 
                error = error_ii;
                x = x_ii;
                x_ind = I;
            end 
        end
    end
    
    X_sparse = zeros(N,1);
    X_sparse(x_ind,:) = x;
    X_error = norm(y - A*X_sparse, 2);
end