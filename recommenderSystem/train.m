function theta = train(Y, R, num_users, num_articles, num_features, lambda)

% Set Initial Parameters (Theta, X)
X = randn(num_articles, num_features);
Theta = randn(num_users, num_features);

initial_parameters = [X(:); Theta(:)];

% Set options for fmincg
options = optimset('GradObj', 'on', 'MaxIter', 100);

% Set Regularization
theta = fmincg (@(t)(cofiCostFunc(t, Y, R, num_users, num_articles, ...
                                num_features, lambda)), ...
                initial_parameters, options);
                                
end