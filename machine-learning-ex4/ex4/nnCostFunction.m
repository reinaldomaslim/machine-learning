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

a1= [ ones(m,1) , X]; %X=5000x400, a1=5000x401

a2=[ones(m,1), sigmoid(a1*Theta1')]; %5000x26
h=sigmoid(a2*Theta2'); %hypothesis=5000x10 

a=1:num_labels;

for i=1:m
    Y=(y(i,1)==a); %10x1
    j=(-Y.*log(h(i,:))-(ones(1, num_labels)-Y).*log(ones(1, num_labels)-h(i,:)))/m;
    J=J+sum(j);
end

J=J+(sum(sum(Theta1(:,2:end).*Theta1(:,2:end)))+sum(sum(Theta2(:,2:end).*Theta2(:,2:end))))*lambda/(2*m);

tri_1=zeros(size(Theta1));
tri_2=zeros(size(Theta2));

for k=1:m
    %1
    a_1=[1; X(k,:)']; %401x1
    z_2=Theta1*a_1; %25x1
    a_2=[1; sigmoid(z_2)]; %26x1
    z_3=Theta2*a_2; %10x1
    a_3=sigmoid(z_3); %10x1
    
    %2 
    Y=(y(k,1)==a)'; %10x1
    del_3=a_3-Y; %10x1
    
    %3
    
    del_2=(Theta2'*del_3).*sigmoidGradient([1; z_2]); %26x10 10x1 26x1 = 26x1
    del_2=del_2(2:end); %25x1
    
    %4
    tri_1=tri_1+del_2*(a_1'); %25x1*1x401=25x401

    tri_2=tri_2+del_3*(a_2'); %10x1*1x26=10x26

end    
        
Theta1_grad=tri_1/m;
Theta2_grad=tri_2/m;

thet1=[zeros(size(Theta1,1),1), Theta1(:, 2:end)];
thet2=[zeros(size(Theta2,1),1), Theta2(:, 2:end)];


Theta1_grad=Theta1_grad+thet1*lambda/m;
Theta2_grad=Theta2_grad+thet2*lambda/m;


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
