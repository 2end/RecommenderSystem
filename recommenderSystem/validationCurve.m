function [lambda_vec, error_train, error_val] = ...
    validationCurve(Y, R, Rval, num_users, num_articles, num_features)

% Selected values of lambda
lambda_vec = [0 0.001 0.003 0.01 0.03 0.1 0.3]';

error_train = zeros(length(lambda_vec), 1);
error_val = zeros(length(lambda_vec), 1);
m = sum(R(:) == 1);
mval = sum(Rval(:) == 1);

for i=1:length(lambda_vec)
  lambda = lambda_vec(i);
  theta = train(Y, R, num_users, num_articles, num_features, lambda);
  error_train(i) = sqrt(cofiCostFunc(theta, Y, R, num_users, num_articles, num_features, 0) / m);
  error_val(i) = sqrt(cofiCostFunc(theta, Y, Rval, num_users, num_articles, num_features, 0) / mval);
end

end