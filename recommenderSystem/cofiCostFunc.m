function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_articles, ...
                                  num_features, lambda)

X = reshape(params(1:num_articles*num_features), num_articles, num_features);
Theta = reshape(params(num_articles*num_features+1:end), ...
                num_users, num_features);

J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

J = 1/2*sum((((X*Theta'-Y).*R).^2)(:)) + lambda/2*sum(Theta(:).^2) + lambda/2*sum(X(:).^2);

X_grad = ((X*Theta'-Y).*R)*Theta+lambda*X;
Theta_grad = ((X*Theta'-Y).*R)'*X+lambda*Theta;

grad = [X_grad(:); Theta_grad(:)];

end



