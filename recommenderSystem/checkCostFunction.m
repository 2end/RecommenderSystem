function checkCostFunction(lambda)
%  lambda is a regularization coefficient  

if ~exist('lambda', 'var') || isempty(lambda)
    lambda = 0;
end

%  X is a 4x3 matrix, containing 3 features of 4 articles
%  Theta is a 5x3 matrix, containing coefficient for 5 users
%
%  Y is a 4x5 matrix, containing ratings (1-5) of 4 articles by 
%  5 users
%
%  R is a 4x5 matrix, where R(i,j) = 1 if and only if user j gave a
%  rating to article i

X_t = rand(4, 3);
Theta_t = rand(5, 3);

Y = X_t * Theta_t';
Y(rand(size(Y)) > 0.5) = 0;
R = zeros(size(Y));
R(Y != 0) = 1;

X = randn(size(X_t));
Theta = randn(size(Theta_t));
num_users = size(Y, 2);
num_articles = size(Y, 1);
num_features = size(Theta_t, 2);

numgrad = computeNumericalGradient( ...
                @(t) cofiCostFunc(t, Y, R, num_users, num_articles, ...
                                num_features, lambda), [X(:); Theta(:)]);

[cost, grad] = cofiCostFunc([X(:); Theta(:)],  Y, R, num_users, ...
                          num_articles, num_features, lambda);

disp([numgrad grad]);
fprintf(['(Left-Numerical Gradient, Right-Analytical Gradient)\n\n']);

diff = norm(numgrad-grad)/norm(numgrad+grad);
fprintf(['The relative difference should be small (less than 1e-9). \n' ...
         '\nRelative Difference: %g\n'], diff);

end