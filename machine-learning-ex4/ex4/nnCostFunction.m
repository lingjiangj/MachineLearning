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

% add bia units to X
a1 = [ones(m,1) X];

%compute a(2)
z1 = Theta1 * a1';
a2 = 1./(1+exp(-z1));

% compute a(3)
a2 = [ones(1,m);a2];
z2 = Theta2 * a2;
a3 = 1./(1+exp(-z2));

%create matrix 10x5000 with each row an index of label
identity = eye(num_labels);
logicalY = identity(y, :); 

%Unregulized cost function
J_unreg=1./m*sum(diag(-logicalY*log(a3)-(1-logicalY)*log(1-a3)));

%Adding regulation
thetaone = Theta1(:,[2:end]);
thetatwo = Theta2(:,[2:end]);
Theta=[transpose(thetaone(:)) transpose(thetatwo(:))]
reg=lambda./(2*m)*sum(Theta.^2)

J=J_unreg+reg %with regulisation 
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

%compute sigma3 and sigma2
sigma3 = a3 - logicalY';
g = a2.*(1-a2);
sigma2 = (Theta2' * sigma3).* g;

%compute delta2 and delta1
delta2 = sigma3 * a2';
delta1 = sigma2(2:end,:) * a1

Theta2_gradunreg=1./m*delta2
Theta1_gradunreg=1./m*delta1

%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
T2 = Theta2
size(T2(:,1))
T2(:,1) = zeros(num_labels,1)
T1 = Theta1
T1(:,1)=zeros(hidden_layer_size,1)
Theta2_grad=Theta2_gradunreg+lambda./m*T2
Theta1_grad=Theta1_gradunreg+lambda./m*T1







% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
