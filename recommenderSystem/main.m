fprintf('\nChecking Gradients (without regularization) ... \n');

checkCostFunction;

fprintf('\nChecking Gradients (with regularization) ... \n');

checkCostFunction(1.5);


articles = loadArticles();

fprintf('\nTraining collaborative filtering...\n');

% Loading article ratings dataset
load('articleRatings.mat');

%  Y is a 60x20 matrix, containing ratings (1-5) of 50 articles by 
%  20 users
%
%  R is a 60x20 matrix, where R(i,j) = 1 if and only if user j gave a
%  rating to article i

%  Normalize Ratings
[Ynorm, Ymean] = normalizeRatings(Y, R);

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
                
save('-v7', 'trainedData.mat', 'X', 'Theta', 'Ymean');

fprintf('Recommender system learning completed.\n');

fprintf('\nProgram paused. Press enter to continue.\n');
pause;
