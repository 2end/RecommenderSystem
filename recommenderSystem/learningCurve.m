function learningCurve(Y, num_users, num_articles, num_features)

% Selected values of lambda
lambda = 0.001;

%Shufle data
Y = Y(:, randperm(size(Y, 2)));

train_coef = 0.6;
val_coef = 1 - train_coef;

num_elems = num_articles * num_users;
train_num_users = train_coef * num_elems;
val_num_users = val_coef * num_elems;

Rval = zeros(num_elems, 1);
for i=1:val_num_users
  Rval(i) = 1;
endfor;
Rval = Rval(randperm(length(Rval)));
Rval = reshape(Rval, num_articles, num_users);

R = zeros(num_articles, num_users);
[k, l] = find(Rval == 0);

error_train = zeros(train_num_users, 1);
error_val = zeros(train_num_users, 1);

for i=1:train_num_users
  R(k(i), l(i)) = 1;
  theta = train(Y, R, num_users, num_articles, num_features, lambda);
  error_train(i) = cofiCostFunc(theta, Y, R, num_users, num_articles, num_features, 0);
  error_val(i) = cofiCostFunc(theta, Y, Rval, num_users, num_articles, num_features, 0);
end

display(error_train)
display(error_val)
close all;

x_values = [1:train_num_users];
plot(x_values, error_train, x_values, error_val);
legend('Train', 'Cross Validation');
xlabel('lambda');
ylabel('Error');

fprintf('lambda\t\tTrain Error\tValidation Error\n');
for i = 1:train_num_users
	fprintf(' %f\t%f\t%f\n', ...
            i, error_train(i), error_val(i));
end

end