function S = update_S(X,M,W)

[D,K] = size(M);
[D,N] = size(X);


tol=1e-3; % regularlizer in case constrained fits are ill conditioned


S = zeros(N,K);
for ii=1:N
   z = repmat(X(:,ii),1,K)-M; % shift i-th point to the origin
   C = z'*W*W'*z;                  % local covariance
   C = C + eye(K,K)*tol*trace(C);                   % regularlization (K>D)
   S(ii,:) = C\ones(K,1);                           % solve Cw=1
   S(ii,:) = S(ii,:)/sum(S(ii,:));                  % enforce sum(w)=1
end






