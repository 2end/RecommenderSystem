function theta = trainContentBased(Y, R, X, lambda)

Theta = randn(size(Y, 2), size(X, 2));

initial_parameters = [Theta(:)];

% Set options for fmincg
options = optimset('GradObj', 'on', 'MaxIter', 100);

% Set Regularization
theta = fmincg (@(t)(cofiCostFuncContentBased(t, Y, R, X, lambda)), ...
                initial_parameters, options);
                
theta = reshape(theta, size(Y, 2), size(X, 2));
                                
end