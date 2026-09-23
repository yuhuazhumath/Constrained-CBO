function problem = quadratic_line()
%QUADRATIC_LINE Quadratic objective with v1+v2-3=0.
d = 2;
problem = struct('name', 'simple_line', 'dimension', d, ...
    'E', @(V) sum(V.^2, 1), 'vstar', [], 'objective_star', NaN, ...
    'metadata', struct('objective', 'squared Euclidean norm'));
g = @(V) sum(V, 1) - 3;
gradg = @(V) ones(size(V));
hessg = @(V) zeros(d, d, size(V, 2));
problem = scalar_constraint(problem, 'line', g, gradg, hessg, [1.5; 1.5]);
end
