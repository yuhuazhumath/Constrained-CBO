function result = run_constrained_cbo_independent_noise(problem,config)
%RUN_CONSTRAINED_CBO_INDEPENDENT_NOISE Algorithm 2 with independent-noise restarts.
config = validate_solver_config(problem,config);
rng(config.seed,'twister');
V = initialize_particles(problem,config);
initial_ensemble = V;
trajectory = trajectory_initialize(problem,config);
trajectory = trajectory_record(trajectory,0,V,problem,config);
best_objective = Inf;
best_consensus = [];
iterations = 0;
restarts = 0;
concentration_events = 0;
incumbent_updates = 0;
timer = tic;
while true
    concentration = trajectory.concentration(iterations+1);
    terminal_event = iterations>=config.max_steps;
    if concentration<=config.concentration_tol
        concentration_events = concentration_events+1;
        % Update the incumbent only at a concentration event.
        [current_consensus,~,~] = compute_consensus(V,problem.E,config.alpha);
        current_objective = problem.E(current_consensus);
        previous_best_objective = best_objective;
        [best_objective,best_consensus,should_stop] = ...
            update_algorithm2_incumbent(current_objective,current_consensus, ...
            best_objective,best_consensus,config.improvement_tol);
        incumbent_updates = incumbent_updates+double( ...
            best_objective<previous_best_objective);
        if terminal_event
            exit_reason = 'max_steps';
            break
        end
        if should_stop
            exit_reason = 'improvement';
            break
        end
        V = V+config.sigma_indep*sqrt(config.gamma)*randn(size(V));
        restarts = restarts+1;
        % Record the restart without advancing the Equation (33) step count.
        trajectory = trajectory_record(trajectory,iterations,V,problem,config);
    end
    if terminal_event
        exit_reason = 'max_steps';
        break
    end
    Z = randn(size(V));
    V = constrained_cbo_step(V,problem,config,Z);
    iterations = iterations+1;
    trajectory = trajectory_record(trajectory,iterations,V,problem,config);
end
elapsed = toc(timer);
if isempty(best_consensus)
    % Fall back to the final consensus if no concentration event occurred.
    result = finalize_solver_result(V,problem,config,trajectory,iterations, ...
        exit_reason,elapsed,config.seed);
    result.output_source = 'final-consensus-fallback';
else
    result = finalize_solver_result(V,problem,config,trajectory,iterations, ...
        exit_reason,elapsed,config.seed,best_consensus);
end
result.initial_ensemble = initial_ensemble;
result.restarts = restarts;
result.restart_count = restarts;
result.concentration_event_count = concentration_events;
result.best_incumbent_update_count = incumbent_updates;
result.best_objective = best_objective;
result.best_consensus = best_consensus;
result.method = 'constrained-cbo-independent-noise';
end
