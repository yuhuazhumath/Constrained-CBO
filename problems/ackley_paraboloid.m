function problem = ackley_paraboloid(d)
%ACKLEY_PARABOLOID Ackley objective with paraboloid equality.
problem = ackley_problem(d, 0.4*ones(d,1), 20, 0.1, 1);
g = @(V) sum(V(1:end-1,:).^2, 1)-V(end,:);
gradg = @(V) [2*V(1:end-1,:); -ones(1,size(V,2))];
hessg = @(V) repmat(diag([2*ones(1,d-1), 0]), 1, 1, size(V,2));
if d == 3
    reference = ackley_references("paraboloid_d3");
    vstar = reference.vstar;
elseif d == 20
    reference = ackley_references("paraboloid_d20");
    vstar = reference.vstar;
else
    vstar = NaN(d,1);
end
problem = scalar_constraint(problem, 'paraboloid', g, gradg, hessg, vstar);
if d == 3 || d == 20
    problem.objective_star = reference.objective_value;
    problem.metadata.reference = reference;
end
end
