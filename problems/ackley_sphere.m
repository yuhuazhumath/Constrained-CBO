function problem = ackley_sphere(d)
%ACKLEY_SPHERE Ackley objective with unit-sphere equality.
problem = ackley_problem(d, 0.4*ones(d,1), 20, 0.1, 1);
g = @(V) sum(V.^2, 1)-1;
gradg = @(V) 2*V;
hessg = @(V) repmat(2*eye(d), 1, 1, size(V, 2));
problem = scalar_constraint(problem, 'sphere', g, gradg, hessg, ...
    ones(d,1)/sqrt(d));
end
