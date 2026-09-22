function problem = quadratic_ellipse()
%QUADRATIC_ELLIPSE Quadratic objective with ellipse equality.
d = 2;
problem = struct('name', 'simple_ellipse', 'dimension', d, ...
    'E', @(V) sum(V.^2, 1), 'vstar', [], 'objective_star', NaN, ...
    'metadata', struct('objective', 'squared Euclidean norm'));
g = @(V) (V(1,:)+1).^2/2 + V(2,:).^2 - 1;
gradg = @(V) [V(1,:)+1; 2*V(2,:)];
hessg = @(V) repmat(diag([1, 2]), 1, 1, size(V, 2));
problem = scalar_constraint(problem, 'ellipse', g, gradg, hessg, ...
    [sqrt(2)-1; 0]);
end
