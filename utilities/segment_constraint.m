function problem = segment_constraint(name, objective, a, b, vstar)
%SEGMENT_CONSTRAINT Direct squared-distance problem for a line segment.
arguments
    name char
    objective function_handle
    a double
    b double
    vstar double = []
end
a = a(:);
b = b(:);
assert(numel(a) == numel(b) && norm(b-a) > 0, ...
    'segment_constraint:InvalidEndpoints', ...
    'Segment endpoints must be distinct and have equal dimension.');
d = numel(a);
u = b-a;
s = u.'*u;
e = u/sqrt(s);
project = @(V) a + u * min(max((u.'*(V-a))/s, 0), 1);

problem = struct();
problem.name = name;
problem.dimension = d;
problem.E = objective;
problem.G = @(V) sum((V-project(V)).^2, 1);
problem.gradG = @(V) 2*(V-project(V));
problem.hessG = @(V) segment_hessian(V, a, u, s, e);
problem.g = @(V) sqrt(max(problem.G(V), 0));
problem.project = project;
problem.vstar = vstar(:);
if isempty(vstar)
    problem.objective_star = NaN;
else
    problem.objective_star = objective(vstar(:));
end
problem.metadata = struct('objective', 'custom', ...
    'constraint', 'line segment squared distance', ...
    'constraint_representation', 'direct G=dist(v,L)^2', ...
    'endpoint_a', a, 'endpoint_b', b, ...
    'switching_hessian_convention', '2I (endpoint side)');
end

function H = segment_hessian(V, a, u, s, e)
d = size(V, 1);
n = size(V, 2);
t = (u.'*(V-a))/s;
H = zeros(d, d, n);
Hendpoint = 2*eye(d);
Hinterior = 2*(eye(d)-e*e.');
for j = 1:n
    % Strict inequalities make both switching hyperplanes use 2I.
    if t(j) > 0 && t(j) < 1
        H(:,:,j) = Hinterior;
    else
        H(:,:,j) = Hendpoint;
    end
end
end
