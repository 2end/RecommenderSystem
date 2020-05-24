function theta = train(Y, R, num_users, num_articles, num_features, lambda)
X = randn(num_articles, num_features);
Theta = zeros(num_users, num_features);

initial_parameters = [X(:); Theta(:)];
options = optimset('GradObj', 'on', 'MaxIter', 100);
theta = fmincg (@(t)(cofiCostFunc(t, Y, R, num_users, num_articles, ...
                                num_features, lambda)), ...
                initial_parameters, options);
                                
end



