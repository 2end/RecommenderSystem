function theta = trainOne(Y, R, X, lambda)

initial_parameters = randn(size(X, 2), 1);

% Set options for fmincg
options = optimset('GradObj', 'on', 'MaxIter', 50);

% Set Regularization
theta = fmincg (@(t)(cofiCostFuncOne(t, Y, R, X, lambda)), ...
                initial_parameters, options);
                
theta = theta';
                                
end