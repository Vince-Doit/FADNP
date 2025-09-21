% minimizing norm(X-HF','pro')^2-gama*trace(F'11'F)
% F:n*c is indicator matrix
% H:d*c clustering assignment
function [Ind,F,H,minObj]= BalancedKmeans_new(X,c,gamma,Iter)
% X : [d x n] where d is the number of feature and n is the number of
% objects
% c: cluster_num
% objects

if nargin<3
    gamma = 0.001;
end
if nargin<4
    Iter = 100;
end
[d,n] = size(X);
% Initialize the indicator matrix F randomly.
% StartInd = randsrc(n,1,1:c);
% F = zeros(n,c);
% for i = 1:n
%     F(i,StartInd(i))=1;
% end

% Initialize with k-means
Y0 = kmeans(X, c);
F=n2nc(Y0);

% iteration
epsilon = 1e-8;
Obj = zeros(Iter,1);
for iter = 1:Iter
    %Fixing F, compute H
    H = X*F/(F'*F);
    cn = sum(F);  % compute the samples of each class.
    Obj(iter) = norm(X-H*F','fro')^2 + gamma*cn*cn';
    G = (n*gamma*eye(n)-gamma*ones(n))*F;
    Q = G+X'*H-1/2*ones(n,1)*ones(1,d)*(H.*H);
    for i = 1:n
        [~, idx_old] = max(F(i, :));
        [~, idx_new] = max(Q(i, :));
        cn(idx_old) = cn(idx_old) - 1;
        F(i, idx_old) = 0;
        cn(idx_new) = cn(idx_new) + 1;
        F(i, idx_new) = 1;
    end
    Obj(iter) = norm(X-H*F', 'fro')^2 +gamma*cn*cn';
    if iter > 2
        Obj_diff = abs(( Obj(iter-1)-Obj(iter))/Obj(iter-1));
        if Obj_diff < epsilon
            break;
        end
    end
    
    if iter == 1
        minObj = Obj(iter);
    elseif minObj > Obj(iter)
        minObj = Obj(iter);
    end
    % stop
    if iter>30
        round_diff = sum(Obj(iter-9:iter-5) -  Obj(iter-4:iter));
        if sum(abs(round_diff)) < epsilon
            break;
        end
    end
end
Ind = nc2n(F);
end