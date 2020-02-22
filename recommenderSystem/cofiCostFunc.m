function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_articles, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_articles, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_articles*num_features), num_articles, num_features);
Theta = reshape(params(num_articles*num_features+1:end), ...
                num_users, num_features);

            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

%  X - num_articles  x num_features matrix of article features
%  Theta - num_users  x num_features matrix of user features
%  Y - num_articles x num_users matrix of user ratings of articles
%  R - num_articles x num_users matrix, where R(i, j) = 1 if the 
%  i-th article was rated by the j-th user
%
%  X_grad - num_articles x num_features matrix, containing the 
%  partial derivatives to each element of X
%  Theta_grad - num_users x num_features matrix, containing the 
%  partial derivatives to each element of Theta
%

J = 1/2*sum((((X*Theta'-Y).*R).^2)(:)) + lambda/2*sum(Theta(:).^2)+lambda/2*sum(X(:).^2);

X_grad = ((X*Theta'-Y).*R)*Theta+lambda*X;
Theta_grad = ((X*Theta'-Y).*R)'*X+lambda*Theta;

grad = [X_grad(:); Theta_grad(:)];

end
