function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

z=theta'*X'; %length of m


[k col]=size(theta);

reg=0;
dreg=zeros(size(theta));

for j=2:k
    reg=reg+lambda*theta(j,1)^2/(2*m);
    dreg(j)=lambda*theta(j,1)/m;
end

for i=1:m
    cost=(-y(i)*log(sigmoid(z(i)))-(1-y(i))*log(1-sigmoid(z(i))))/m;
    J=J+cost;
    n=(sigmoid(z(i))-y(i))*X(i,:)/m; %1xn
    grad=grad+n';
end

grad=grad+dreg;
J=J+reg;




% =============================================================

end
