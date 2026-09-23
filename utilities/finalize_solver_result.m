function result = finalize_solver_result(V,problem,config,trajectory,iterations,exit_reason,elapsed,seed,output_consensus)
%FINALIZE_SOLVER_RESULT Evaluate metrics at the solver's output consensus.
[final_ensemble_consensus,weights] = compute_consensus( ...
    V,problem.E,config.alpha);
if nargin<9 || isempty(output_consensus)
    % Algorithm 1: consensus of the final ensemble.
    v_out = final_ensemble_consensus;
    output_source = 'fresh-final-ensemble-consensus';
else
    % Algorithm 2: the best concentrated consensus accepted over restarts.
    v_out = output_consensus;
    output_source = 'best-concentrated-consensus';
end
metrics = compute_metrics(problem,v_out,config);
result = metrics;
result.seed = seed;
result.success = metrics.success;
result.v_out = v_out;
result.iterations = iterations;
result.exit_reason = exit_reason;
result.runtime = elapsed;
result.final_ensemble = V;
result.final_weights = weights;
result.final_ensemble_consensus = final_ensemble_consensus;
result.output_source = output_source;
result.trajectory = trajectory;
result.active_mask = trajectory.active_mask;
result.config = config;
result.problem_name = problem.name;
end
