function H = Hess_eg3(gd_g,g,hess_g)
dim = size(gd_g,1);
H = zeros(size(hess_g));
for i = 1:size(g,2)
    H((i-1)*dim+1:i*dim,(i-1)*dim+1:i*dim) = 2*gd_g(:,i)*gd_g(:,i)' + 2*g(i)*hess_g((i-1)*dim+1:i*dim,(i-1)*dim+1:i*dim);
end