function f = obj_fcn(W)  
global b; global A; global a;
% diff = W - w_star;
% f = -A*exp(-a*sqrt(b^2/size(W,1))*sum(diff.^2,1))...
%     - exp(sum(cos(2*pi*b*diff),1)/size(W,1))+exp(1)+B;


dim = size(W,1);
diff = W - 0.4*ones(dim,1);
%f = 1/dim*sum(diff.^2 - A*cos(2*pi*diff) + A,1) + C;

b = 1; A = 20; a = 0.1;
f = -A*exp(-a*sqrt(b^2/dim*sum(diff.^2))) - exp(1/dim*sum(cos(2*pi*b*diff))) + A + exp(1);