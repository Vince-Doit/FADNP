% minimizing norm(X-HF','pro')^2-gama*trace(F'11'F)
% F:n*k is indicator matrix         H:d*n clustering assignment
function [centroid,clusterAssent ]= BalancedKmeans_chen(data,k,gamma,Iter)
% Data : n*d
% K: cluster_num

if nargin<3
%     gamma = 0.001;
    gamma = 0.001;
end
if nargin<4
    Iter = 100;
end
X = data';
[d,n] = size(X);
% Initialize the indicator matrix F randomly.
StartInd = randsrc(n,1,1:k);
F = zeros(n,k);
for i = 1:n
    F(i,StartInd(i))=1;
end

% Y0 = kmeans(data, k, 'emptyaction','drop', 'Replicates', 5);
% F=n2nc(Y0);

% iteration
epsilon = 1e-8;


Obj = zeros(Iter,1);
S=zeros(n,k);
for iter=1:Iter
    % Fixing F,compute H
    H = X*F/(F'*F);
    
    %Fixing H,update F
    %update F row by row
    
    for itr2=1:10
        F_old=F;
        sumF=sum(F,1);
        S(:,:)=0;
        F(:,:)=0;
        for i=1:n
            idx=find(F_old(i,:)==1);
            sumF(idx)=sumF(idx)-1;
            for l=1:k
                S(i,l)=sum((X(:,i)-H(:,l)).^2);
                S(i,l)=S(i,l)+2*gamma*(sumF(l)-F_old(i,l));
            end

            [~,idx]=min(S(i,:),[],2);
            F(i,idx)=1;
            sumF(idx)=sumF(idx)+1;
        end
        if sum(sum(abs(F_old-F)))==0
            break;
        end
    end 
    
    Obj(iter) = norm(X-H*F','fro')^2 + gamma*sum(sum(F,1).^2);
    if iter>2
        Obj_diff = abs(( Obj(iter-1)-Obj(iter))/Obj(iter-1));
        if Obj_diff < epsilon
            break;
        end
    end
end
centroid = zeros(k,d);
clusterAssent = zeros(n,2);
idx = nc2n(F);
clusterAssent(:,1) = idx;
% for  i= 1:k
%     curX = data(find(idx == j),:);
%     centroid(j,:) =mean( curX);
% end
for  i= 1:k
    cidx=find(idx == i);
    curX = data(cidx,:);
    centroid(i,:) =mean( curX);
    diff = curX - repmat(centroid(i,:), size(curX,1), 1);
    clusterAssent(cidx, 2) = sum(diff.*diff, 2);
end
end

