function f = obj_fcn_0(W)  
global w_star; global b; global A; global a;global B;
diff = W - w_star;
f = -A*exp(-a*sqrt(b^2/size(W,1))*sum(diff.^2,1))...
    - exp(sum(cos(2*pi*b*diff),1)/size(W,1))+exp(1)+B;


% dim = size(W,1);
% C = 0;
% diff = W- w_star;
% f = 1/dim*sum(diff.^2 - A*cos(2*pi*diff) + A,1) + C;