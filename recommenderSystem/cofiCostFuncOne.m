function [J, grad] = cofiCostFuncOne(params, Y, R, X, lambda)

Theta = params';
            
J = 0;
Theta_grad = zeros(size(Theta));

J = 1/2*sum((((X*Theta'-Y).*R).^2)(:)) + lambda/2*sum(Theta(:).^2);

Theta_grad = ((X*Theta'-Y).*R)'*X+lambda*Theta;
grad = Theta_grad';

end
