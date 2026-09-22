function problem = ackley_ball()
%ACKLEY_BALL Ackley on an embedded 18-ball in R^20.
% G is squared distance to K. At ||head||=R the head block of hessG uses
% the interior-side convention (zero); the two tail directions always use 2I.
d = 20;
m = 18;
radius = 0.5;
base = ackley_problem(d, 0.4*ones(d,1), 20, 0.1, 1);
project = @(V) project_embedded_ball(V, m, radius);
reference = ackley_references("embedded_ball_d20");
problem = base;
problem.name = 'ackley_embedded_ball';
problem.G = @(V) sum((V-project(V)).^2,1);
problem.gradG = @(V) 2*(V-project(V));
problem.hessG = @(V) embedded_ball_hessian(V,m,radius);
problem.g = @(V) sqrt(max(problem.G(V),0));
problem.project = project;
problem.vstar = reference.vstar;
problem.objective_star = reference.objective_value;
problem.metadata.constraint = 'embedded 18-ball squared distance';
problem.metadata.constraint_representation = 'direct G=dist(v,K)^2';
problem.metadata.radius = radius;
problem.metadata.boundary_hessian_convention = 'interior-side head Hessian';
problem.metadata.reference = reference;
end

function P = project_embedded_ball(V,m,radius)
head = V(1:m,:);
r = sqrt(sum(head.^2,1));
scale = ones(1,size(V,2));
outside = r>radius;
scale(outside) = radius./r(outside);
P = [head.*scale; zeros(size(V,1)-m,size(V,2))];
end

function H = embedded_ball_hessian(V,m,radius)
d = size(V,1);
n = size(V,2);
head = V(1:m,:);
r = sqrt(sum(head.^2,1));
H = zeros(d,d,n);
tail = d-m;
H(m+1:end,m+1:end,:) = repmat(2*eye(tail),1,1,n);
Im = eye(m);
for j = 1:n
    % At the boundary, retain the interior-side zero head block.
    if r(j)>radius
        z = head(:,j);
        H(1:m,1:m,j) = 2*((1-radius/r(j))*Im ...
            + radius/(r(j)^3)*(z*z.'));
    end
end
end
