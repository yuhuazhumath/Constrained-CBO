function result = run_quadratic_penalty_cbo(problem,config)
%RUN_QUADRATIC_PENALTY_CBO Fixed smooth quadratic-penalty CBO baseline.
config = validate_solver_config(problem,config);
rng(config.seed,'twister');
V = initialize_particles(problem,config);
initial_ensemble = V;
trajectory = trajectory_initialize(problem,config);
objective = @(X) problem.E(X)+problem.G(X)/config.epsilon;
trajectory = trajectory_record(trajectory,0,V,problem,config,objective);
iterations = 0;
exit_reason = 'max_steps';
timer = tic;
while iterations<config.max_steps
    if trajectory.concentration(iterations+1)<=config.concentration_tol
        exit_reason = 'concentration';
        break
    end
    [v_alpha,~,~] = quadratic_penalty_consensus(V,problem,config);
    diff = V-v_alpha;
    Z = randn(size(V));
    V = V-config.lambda*config.gamma*diff ...
        -config.sigma*sqrt(config.gamma)*diff.*Z;
    iterations = iterations+1;
    trajectory = trajectory_record(trajectory,iterations,V,problem,config,objective);
end
elapsed = toc(timer);
% Form the final consensus with penalized weights; evaluate metrics using E and G.
[v_out,weights] = quadratic_penalty_consensus(V,problem,config);
metrics = compute_metrics(problem,v_out,config);
result = metrics;
result.seed = config.seed;
result.iterations = iterations;
result.exit_reason = exit_reason;
result.runtime = elapsed;
result.initial_ensemble = initial_ensemble;
result.final_ensemble = V;
result.final_weights = weights;
result.final_ensemble_consensus = v_out;
result.output_source = 'fresh-final-quadratic-penalty-consensus';
result.trajectory = trajectory;
result.active_mask = trajectory.active_mask;
result.config = config;
result.problem_name = problem.name;
result.method = 'quadratic-penalty-cbo';
end
