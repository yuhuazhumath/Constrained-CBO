function result = run_constrained_cbo(problem,config)
%RUN_CONSTRAINED_CBO Run Algorithm 1 using the shared Equation (33) step.
config = validate_solver_config(problem,config);
rng(config.seed,'twister');
V = initialize_particles(problem,config);
initial_ensemble = V;
trajectory = trajectory_initialize(problem,config);
trajectory = trajectory_record(trajectory,0,V,problem,config);
iterations = 0;
exit_reason = 'max_steps';
timer = tic;
while iterations<config.max_steps
    current_concentration = trajectory.concentration(iterations+1);
    if current_concentration<=config.concentration_tol
        exit_reason = 'concentration';
        break
    end
    Z = randn(size(V));
    V = constrained_cbo_step(V,problem,config,Z);
    iterations = iterations+1;
    trajectory = trajectory_record(trajectory,iterations,V,problem,config);
end
elapsed = toc(timer);
result = finalize_solver_result(V,problem,config,trajectory,iterations, ...
    exit_reason,elapsed,config.seed);
result.initial_ensemble = initial_ensemble;
result.method = 'constrained-cbo-equation-33';
end
