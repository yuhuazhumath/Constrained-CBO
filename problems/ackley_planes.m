function problem = ackley_planes()
%ACKLEY_PLANES Three-dimensional two-constraint case.
d = 3;
problem = ackley_problem(d, 0.4*ones(d,1), 20, 0.1, 1);
g = @(V) [sum(V,1)-1; 2*sum(V(1:end-1,:),1)-0.5*V(end,:)-0.5];
B = [1,1,1; 2,2,-0.5];
problem.name = 'ackley_two_affine';
problem.g = g;
problem.G = @(V) sum(g(V).^2, 1);
problem.gradG = @(V) 2*B.'*g(V);
problem.hessG = @(V) repmat(2*(B.'*B), 1, 1, size(V,2));
problem.vstar = [0.2;0.2;0.6];
problem.objective_star = problem.E(problem.vstar);
problem.metadata.constraint = 'two affine equalities';
problem.metadata.constraint_representation = 'vector g; G=||g||_2^2';
end
