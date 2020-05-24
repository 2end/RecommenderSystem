function validationCurve(Y, R, Rval, Rtest, num_users, num_articles, num_features)

% Selected values of lambda
lambda_vec = [0 0.001 0.003 0.01 0.03 0.1 0.3]';
lambda_test = 0.001;

error_train = zeros(length(lambda_vec), 1);
error_val = zeros(length(lambda_vec), 1);
m = sum(R(:) == 1);
mval = sum(Rval(:) == 1);
mtest = sum(Rtest(:) == 1);

for i=1:length(lambda_vec)
  lambda = lambda_vec(i);
  theta = train(Y, R, num_users, num_articles, num_features, lambda);
  error_train(i) = (cofiCostFunc(theta, Y, R, num_users, num_articles, num_features, 0) / m);
  error_val(i) = (cofiCostFunc(theta, Y, Rval, num_users, num_articles, num_features, 0) / mval);
end

theta = train(Y, R, num_users, num_articles, num_features, lambda_test);
error_test = (cofiCostFunc(theta, Y, Rtest, num_users, num_articles, num_features, 0) / mtest);

display(error_train)
display(error_val)
close all;
plot(lambda_vec, error_train, lambda_vec, error_val);
legend('Train', 'Cross Validation');
xlabel('lambda');
ylabel('Error');

fprintf('lambda\t\tTrain Error\tValidation Error\n');
for i = 1:length(lambda_vec)
	fprintf(' %f\t%f\t%f\n', ...
            lambda_vec(i), error_train(i), error_val(i));
end

display(lambda_test)
display(error_test)

end