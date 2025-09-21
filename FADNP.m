function [W,S_divide,Obj,M_init] =FADNP(X,Y,d,K)
% X is the data matrix (DxN)
% Y is the label vector N x 1
% d is the number of selected features
% K is the number of anchor points in each class
% v is the number of nonzero elements of reconstruction coefficients.
% Obj contains the objective function values
% S_divide contains the reconstruction coefficients.
% W is the feature selection matrix 



maxIter = 60;
thresh = 1e-5;

[n_fea,n_sample] = size(X);
unique_class =  unique(Y);
n_class = length(unique_class);

n_each_class = zeros(1,n_class); 

X_divide = cell(1,n_class);
S_divide = cell(1,n_class);
M_divide = cell(1,n_class);



%%% set X_divide, initialize S_divide 

  for i = 1:n_class

        ind = find(Y == unique_class(i));
        X_divide{i} = X(:,ind);

        n_each_class(i) = length(ind);
        
       if n_each_class(i)<K
  	    error('The number of abchor points is too large');
       end

       [~,C] = kmeans(X_divide{i}',K);
       M_divide{i} = C';
%       [~,M_divide{i} ]= BalancedKmeans_chen(X_divide{i},K,1e-3,50);
      S_divide{i}  = Initalize_S(X_divide{i},M_divide{i}); 
  end
  
  M_init = M_divide; 

  if sum(n_each_class)~=n_sample
  	error('n_each_class is default');
  end



Obj=zeros(1,maxIter);
St  = X*X';
invSt = inv(St);
for it = 1:maxIter

  % solve W 
  P = zeros(n_fea,n_fea);
  for i = 1:n_class
      P = P + (X_divide{i}-M_divide{i}*S_divide{i}')*(X_divide{i}'-S_divide{i}*M_divide{i}');
  end
  
%   [idx,element] =  sort(diag(P)); 
%   W = zeros(n_fea,d);
%   for i = 1:d
%     W(element(i),i) = 1;
%   end 

%   I1 = eye(n_fea);
%   Te = invSt*P+I1;
%   W = eig1(Te,d,0,0);
%   W = real(W);
%   W=W*diag(1./sqrt(diag(W'*St*W)));

  W = eig1(P,d,0,0);

      %The objective function 
      Obj(it) = trace(W'*P*W);
       if (it>1)
       	 if abs(Obj(it)-Obj(it-1))<thresh
       		break;
       	 end
       end
       

       % update S
       for i = 1:n_class 
           S_divide{i}  = update_S(X_divide{i},M_divide{i},W);
       end
       
       % update M
       I2 = eye(K);
       M_divide{i} = X_divide{i}*S_divide{i}/(S_divide{i}'*S_divide{i}+I2);

       

end

Obj = Obj(1:it-1);

 

 