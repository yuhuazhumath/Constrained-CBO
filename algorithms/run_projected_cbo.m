function result = run_projected_cbo(problem,config)
%RUN_PROJECTED_CBO Sphere-constrained KV-CBO baseline.
% The method requires feasible initialization. Experiment runners normalize
% corresponding base samples generated from the shared repetition seed.
config = validate_solver_config(problem,config);
rng(config.seed,'twister');
V = initialize_particles(problem,config);
V = normalize_to_sphere(V);
initial_ensemble = V;
trajectory = trajectory_initialize(problem,config);
trajectory = trajectory_record(trajectory,0,V,problem,config);
iterations = 0;
exit_reason = 'max_steps';
timer = tic;
while iterations<config.max_steps
    if trajectory.concentration(iterations+1)<=config.concentration_tol
        exit_reason = 'concentration';
        break
    end
    V = projected_cbo_step(V,problem,config,randn(size(V)));
    iterations = iterations+1;
    trajectory = trajectory_record(trajectory,iterations,V,problem,config);
end
elapsed = toc(timer);
result = finalize_solver_result(V,problem,config,trajectory,iterations, ...
    exit_reason,elapsed,config.seed);
result.initial_ensemble = initial_ensemble;
result.method = 'projected-cbo-fhps-algorithm-1';
end
