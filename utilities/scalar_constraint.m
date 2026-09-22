function problem = scalar_constraint(problem, name, g, gradg, hessg, vstar)
%SCALAR_CONSTRAINT Attach a smooth scalar g with G=g^2.
arguments
    problem struct
    name char
    g function_handle
    gradg function_handle
    hessg function_handle
    vstar double
end

d = problem.dimension;
problem.name = sprintf('%s_%s', problem.name, name);
problem.g = g;
problem.G = @(V) g(V).^2;
problem.gradG = @(V) 2 .* g(V) .* gradg(V);
problem.hessG = @(V) scalar_hessG(V, g, gradg, hessg, d);
problem.vstar = vstar(:);
if any(isnan(problem.vstar))
    problem.objective_star = NaN;
else
    problem.objective_star = problem.E(problem.vstar);
end
problem.metadata.constraint = name;
problem.metadata.constraint_representation = 'smooth scalar g; G=g^2';
end

function H = scalar_hessG(V, g, gradg, hessg, d)
n = size(V, 2);
gv = g(V);
J = gradg(V);
Hg = hessg(V);
H = zeros(d, d, n);
for j = 1:n
    H(:,:,j) = 2 * (J(:,j)*J(:,j).' + gv(j)*Hg(:,:,j));
end
end
