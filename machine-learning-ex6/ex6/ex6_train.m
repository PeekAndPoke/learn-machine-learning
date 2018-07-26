%% Initialization
clear ; close all; clc

%% ========== Part 7: Training SVM with RBF Kernel (Dataset 3) ==========

%  This is a different dataset that you can use to experiment with. Try
%  different values of C and sigma here.
% 


% Load from ex6data3: 
% You will have X, y in your environment
load('ex6data3.mat');



% Try different SVM Parameters here
[C, sigma] = dataset3Params(X, y, Xval, yval);

bestSettings    = [0 0];
bestPerformance = 0;

for C = [0.01  0.03  0.1  0.3 1 3 10 30];
  
  for sigma = [0.01 0.03 0.1 0.3 1 3 10 30];

    disp("====================================================================");
    disp(["training witth C=" num2str(C) " sigma=" num2str(sigma)]);
    
    % Train the SVM
    model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
    visualizeBoundary(X, y, model);

    prediction = svmPredict(model, Xval);
    result = mean(double(prediction == yval)) * 100;

    disp(["C=" num2str(C) " sigma=" num2str(sigma) " result=" num2str(result)]);
    disp("")
    
    if (result > bestPerformance)
      disp(["NEW RECORD: " num2str(bestPerformance) " -> " num2str(result)])
      disp("")
      
      bestSettings    = [C sigma];
      bestPerformance = result;
    end    
    
  end
  
end

disp("Best settings");
disp(bestSettings);