function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


% recode the y vec from 0 .. 10 to [1 0 0 0 0 0 0 0 0 0] .. [0 0 0 0 0 0 0 0 0 10]
yv=[1:num_labels] == y;


% ==============================================================================
% PART 1 - Forward propagation
% ==============================================================================

% ------------------------------------------------------------------------------
% Calculate cost - evaluate the hypothesis

%{

% using a for loop

jAccu = 0;
  
for i = 1:m

  % get inputs and outputs of training data
  input  = X(i,:);
  output = yv(i, :);
  
  % apply forward propagation
  a1 = [1, input];
  
  z2 = a1 * Theta1';
  a2 = [1, sigmoid(z2)];
  
  z3 = a2 * Theta2';
  a3 = sigmoid(z3);
  
  jLocal = sum( (-output .* log(a3)) - ((1 - output) .* log(1 - a3)));
  jAccu  = jAccu + jLocal;

end

J = (1/m) * jAccu;
%}


% ------------------------------------------------------------------------------
% using vectorized approach

a1 = [ones(m, 1), X]; % injecting bias at the front

z2 = a1 * Theta1';
a2 = [ones(m, 1), sigmoid(z2)]; % injecting bias at the front

z3 = a2 * Theta2';
a3 = sigmoid(z3);

J = (1/m) * sum(sum(-yv .* log(a3) - (1 - yv) .* log(1 - a3)));


% ------------------------------------------------------------------------------
% add regularization

% strip the bias from the thetas
Theta1Raw = Theta1(:, 2:end);
Theta2Raw = Theta2(:, 2:end);

% Note: sumsq(x) is equivalent to sum(x .^ 2);
reg  = (lambda / (2*m)) * (sum(sumsq(Theta1Raw)) + sum(sumsq(Theta2Raw)));
J   += reg;


% ==============================================================================
% PART 2 - Back propagation
% ==============================================================================

d3 = a3 - yv;                                                 % has same dimensions as a3
d2 = (d3*Theta2) .* [ones(size(z2,1),1) sigmoidGradient(z2)]; % has same dimensions as a2

D1 = d2(:,2:end)' * a1; % has same dimensions as Theta1
D2 = d3' * a2;          % has same dimensions as Theta2

Theta1_grad = (1/m) * D1;
Theta2_grad = (1/m) * D2;

% REGULARIZATION OF THE GRADIENT

Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + (lambda/m)*(Theta1(:,2:end));
Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + (lambda/m)*(Theta2(:,2:end));

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
