function problem = quadratic_segment()
%QUADRATIC_SEGMENT Quadratic objective constrained to a segment.
% At either projection-switching hyperplane, hessG uses the endpoint value 2I.
a = [1.6; 0.2; 0.4];
b = [-0.3; -0.7; 0.5];
problem = segment_constraint('simple_segment', @(V) sum(V.^2, 1), a, b, []);
problem.vstar = problem.project(zeros(3,1));
problem.objective_star = problem.E(problem.vstar);
end
