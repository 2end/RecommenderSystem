fprintf('\nChecking Gradients (without regularization) ... \n');

checkCostFunction;

fprintf('\nChecking Gradients (with regularization) ... \n');

checkCostFunction(1.5);


articles = loadArticles();

%  Initialize my ratings to articles
my_ratings = zeros(50, 1);

my_ratings(1) = 5;
my_ratings(2) = 5;
my_ratings(3) = 5;
my_ratings(4)= 5;
my_ratings(5) = 5;
my_ratings(11)= 5;
my_ratings(12)= 5;
my_ratings(13) = 5;
my_ratings(14) = 5;
my_ratings(15) = 5;
my_ratings(41)= 1;
my_ratings(42)= 1;
my_ratings(43)= 1;
my_ratings(44)= 1;
my_ratings(45)= 1;

fprintf('\n\nMy ratings:\n');
for i = 1:length(my_ratings)
    if my_ratings(i) > 0 
        fprintf('Rated %d for %s\n', my_ratings(i), ...
                 articles{i});
    end
end

fprintf('\nTraining collaborative filtering...\n');

% Loading article ratings dataset
load('articleRatings.mat');
load('Rs.mat');

%  Y is a 50x20 matrix, containing ratings (1-5) of 50 articles by 
%  20 users
%
%  R is a 50x20 matrix, where R(i,j) = 1 if and only if user j gave a
%  rating to article i

%  Add my ratings to the data matrix
%Y = [my_ratings Y];
%R = [(my_ratings != 0) R];
%Rval = [(my_ratings != 0) Rval];
%Rtest = [(my_ratings != 0) Rtest];

%  Normalize Ratings
[Ynorm, Ymean] = normalizeRatings(Y, ones(size(R)));

%  Useful Values
num_users = size(Y, 2);
num_articles = size(Y, 1);
num_features = 5;

lambda = 0.001;         
theta = train(Ynorm, R, num_users, num_articles, num_features, lambda);

% Unfold the returned theta back into U and W
X = reshape(theta(1:num_articles*num_features), num_articles, num_features);
Theta = reshape(theta(num_articles*num_features+1:end), ...
                num_users, num_features);

fprintf('Recommender system learning completed.\n');

fprintf('\nProgram paused. Press enter to continue.\n');
pause;

%  Computing the predictions matrix.
%p = X * Theta';
%my_predictions = p(:,1) + Ymean;

%articles = loadArticles();

%[r, ix] = sort(my_predictions, 'descend');
%fprintf('\nTop recommendations for me:\n');
%for i=1:10
%    j = ix(i);
%    fprintf('Predicting rating %.1f for article %s\n', my_predictions(j), ...
%            articles{j});
%end

%fprintf('\n\nOriginal ratings provided:\n');
%for i = 1:length(my_ratings)
%    if my_ratings(i) > 0 
%        fprintf('Rated %d for %s\n', my_ratings(i), ...
%                 articles{i});
%    end
%end

% Validation for Selecting Lambda
[lambda_vec, error_train, error_val] = ...
    validationCurve(Ynorm, R, Rval, num_users, num_articles, num_features);

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
