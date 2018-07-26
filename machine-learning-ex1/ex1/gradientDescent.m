function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %


    % calc hypothesis for each example
    hypo = X * theta;
    % calc dist from expected output for each example
    dist = hypo - y;
    
    % calc the gradient by adding up the contributions of each example
    grad = (alpha / m) * X' * dist;
    
    % calc new theta
    theta = theta - grad; 
    
%    theta = [ 
%      theta(1) - (alpha / m) * xThetaMinusYTransp * X(:,1);  
%      theta(2) - (alpha / m) * xThetaMinusYTransp * X(:,2) 
%    ];



    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

    
end

end
