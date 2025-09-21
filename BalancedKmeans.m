% minimizing norm(X-HF','pro')^2-gama*trace(F'11'F)
% F:n*k is indicator matrix
% H:d*n clustering assignment
function [Ind,F,H,minObj,obj]= BalancedKmeans(X,c,gamma,Iter)
% X : [d x n] where d is the nnumber of feature and n is the number of
% objects
% c: cluster_num
% objects

if nargin<3
    gamma = 0.001;
end
if nargin<4
    Iter = 100;
end
[~,n] = size(X);
% Initialize the indicator matrix F randomly.
% StartInd = randsrc(n,1,1:k);
% F = zeros(n,k);
% for i = 1:n
%     F(i,StartInd(i))=1;
% end

% Initialize with k-means
Y0 = kmeans(X', c, 'emptyaction','drop');
F=n2nc(Y0);

% iteration
epsilon = 1e-8;
Obj = zeros(Iter,1);
for iter=1:Iter
    % Fixing F,compute H
    H = X*F/(F'*F);
    cn=sum(F);
    
    Obj(iter) = norm(X-H*F','fro')^2 + gamma*cn*cn';
    
    
    [~,label] = max(bsxfun(@minus,center'*X,0.5*sum(center.^2,1)'),[],1); % assign labels
    obj = zeros(n,c);
    for i = 1:n
        for j = 1:c
            [~, cidx] = max(F(i,:));
            cn(cidx)=cn(cidx)-1;
            F(i,cidx) = 0;
            cn(j)=cn(j)+1;
            F(i,j) = 1;
            obj(i,j) = norm(X(:,i)-H*F(i,:)','fro')^2 + gamma*cn*cn';
        end
        
        [~, cidx] = max(F(i,:));
        F(i,cidx) = 0;
        cn(cidx) = cn(cidx) - 1;
        [~,ind] = min(obj(i,:));
        F(i,ind) = 1;
        cn(ind) = cn(ind) + 1;
    end
    Obj(iter) = norm(X-H*F','fro')^2 + gamma*cn*cn';
    if iter>2
        Obj_diff = abs(( Obj(iter-1)-Obj(iter))/Obj(iter-1));
        if Obj_diff < epsilon
            break;
        end
    end
    
    if iter==1
        minObj=Obj(iter);
    elseif minObj>Obj(iter)
        minObj= Obj(iter);
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

% centroid = zeros(d,k);
% clusterAssent = zeros(2,n);
% Ind = nc2n(F);
% clusterAssent(:,1) = Ind;
% % for  i= 1:k
% %     curX = data(find(idx == j),:);
% %     centroid(j,:) =mean( curX);
% % end
% for  i= 1:k
%     cidx=find(Ind == i);
%     curX = X(:,cidx);
%     centroid(:,i) =mean(curX);
%     diff = curX - repmat(centroid(i,:), [size(curX,1) 1]);
%     clusterAssent(2, cidx) = sum(diff.*diff, 2);
% end
end

