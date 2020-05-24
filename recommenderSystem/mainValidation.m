load('articleRatings.mat');
load('RsValidation.mat');
  
%  Useful Values
num_users = size(Y, 2);
num_articles = size(Y, 1);
num_features = 5;

%  Normalize Ratings
[Ynorm, Ymean] = normalizeRatings(Y, ones(size(Y)));
  
% Validation for Selecting Lambda
%validationCurve(Ynorm, R, Rval, Rtest, num_users, num_articles, num_features);

% Learning curve
learningCurve(Ynorm, R, Rval, num_users, num_articles, num_features);
