N=100; ss=3;

myVector = [1,0,2,0,0,3,4];

N = 256; K = 5;

tol = 10e-6;
num_measurements = 10:10:100;

q=randsample(1:N,K)
% for M = num_measurements
%     count = 0;
%     for ii = 1:100
%         x=zeros(N,1); q=randsample(1:N,K); x(q(1:K))=randn(K,1);
%         A=randn(M, N); A=orth(A')';
% 
%         y = A*x;
%         x_sp = SP(A,y,K);
%         error_ii = norm(x_sp - x,2);
%         if error_ii <= tol
%             count = count+1;
%         end
%     end
%     count;
% end
% 
% 
% 
