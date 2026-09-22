function problem = ackley_problem(d, vhat, A, a, b)
%ACKLEY_PROBLEM Create an unconstrained canonical Ackley problem.
arguments
    d (1,1) double {mustBeInteger, mustBePositive}
    vhat double
    A (1,1) double = 20
    a (1,1) double = 0.1
    b (1,1) double = 1
end
assert(numel(vhat) == d, 'ackley_problem:DimensionMismatch', ...
    'vhat must have d entries.');

parameters = struct('A', A, 'a', a, 'b', b, 'vhat', vhat(:));
problem = struct();
problem.name = sprintf('ackley_d%d', d);
problem.dimension = d;
problem.E = @(V) ackley_objective(V, parameters);
problem.G = @(V) zeros(1, size(V, 2));
problem.gradG = @(V) zeros(size(V));
problem.hessG = @(V) zeros(d, d, size(V, 2));
problem.g = @(V) zeros(0, size(V, 2));
problem.vstar = vhat(:);
problem.objective_star = 0;
problem.metadata = struct('objective', 'Ackley', 'ackley', parameters, ...
    'constraint', 'none');
end
