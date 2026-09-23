function result = augment_revision_result(result,problem,initial_seed,solver_seed,repetition)
%AUGMENT_REVISION_RESULT Add analysis aliases and recorded pairing metadata.
result.initial_seed = initial_seed;
result.stochastic_seed = solver_seed;
result.solver_seed = solver_seed;
result.repetition = repetition;
result.alpha = result.config.alpha;
result.epsilon = result.config.epsilon;
result.alpha_over_epsilon = result.alpha/result.epsilon;
result.objective_gap = abs(result.objective_value-problem.objective_star);
result.runtime_seconds = result.runtime;
finite_scalars = [result.objective_value,result.objective_gap, ...
    result.G_value,result.feasibility_error,result.distance_to_vstar, ...
    result.iterations,result.runtime_seconds];
result.all_finite = all(isfinite(result.v_out),'all') ...
    && all(isfinite(result.final_ensemble),'all') ...
    && all(isfinite(finite_scalars));
end
