function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
numberOfTrainingExamples = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

summation = 0;
guesses = sigmoid(theta' * X');
numberOfFeatures = size(X, 2); 

for i = 1:numberOfTrainingExamples
  summation = summation + (y(i) * log(guesses(i)) + (1 - y(i)) * log(1 -  guesses(i))); 
end

J = -summation / numberOfTrainingExamples;

for j = 1:numberOfFeatures
  summation = 0;
  for i = 1:numberOfTrainingExamples
    summation = summation + (guesses(i) - y(i)) * X(i,j);
  end
  grad(j) = summation/numberOfTrainingExamples;
end

% =============================================================

end
