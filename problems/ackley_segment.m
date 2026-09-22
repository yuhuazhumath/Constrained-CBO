function problem = ackley_segment()
%ACKLEY_SEGMENT Manuscript Ackley Case 1 in R^3.
base = ackley_problem(3, 0.4*ones(3,1), 20, 0.1, 1);
a = [0.2;0.5;0.7];
b = [0.5;0.2;0.5];
reference = ackley_references("segment_d3");
problem = segment_constraint('ackley_segment', base.E, a, b, ...
    reference.vstar);
problem.objective_star = reference.objective_value;
problem.metadata.objective = 'Ackley';
problem.metadata.ackley = base.metadata.ackley;
problem.metadata.reference = reference;
end
