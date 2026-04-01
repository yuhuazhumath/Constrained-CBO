function f = obj_fcn_penalty(W,type) 
if type == 1
constrain_fcn = @(W)    sum(W.^2,1) - 1;
end

if type == 2
constrain_fcn = @(W)    W(1,:).^2 - W(2,:);
end

global w_star; global b; global A; global a;global B; global eps;
diff = W - w_star;
f = -A*exp(-a*sqrt(b^2/size(W,1))*sum(diff.^2,1))...
    - exp(sum(cos(2*pi*b*diff),1)/size(W,1))+exp(1)+B;
g = constrain_fcn(W);
f = f+1/eps*g.^2;