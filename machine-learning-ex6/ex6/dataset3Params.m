function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C_set = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
sigma_set = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];



% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
 

for i=1:length(C_set)
    for j=1:length(sigma_set)
        
    model= svmTrain(X, y, C_set(i), @(x1, x2) gaussianKernel(x1, x2, sigma_set(j))); 
    
    predictions = svmPredict(model, Xval);
    
    if (j==1 && i==1)
        m=i;
        n=j;
        min=mean(double(predictions ~= yval));
    end
     
    if mean(double(predictions ~= yval))<min
        min=mean(double(predictions ~= yval));
        m=i;
        n=j;
    end
         
    end
end
 

C=C_set(m);
sigma=sigma_set(n);



% =========================================================================

end
