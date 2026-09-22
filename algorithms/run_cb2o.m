function result = run_cb2o(problem,config)
%RUN_CB2O Run derivative-free Consensus-Based Bi-Level Optimization.
% Equations (5.1)-(5.3) and Algorithm 1 of the CB2O reference use:
%   G_CB2O = problem.E (upper objective minimized over feasibility)
%   L_CB2O = problem.G (lower nonnegative constraint potential)
% config.epsilon is unused by CB2O.

config = validate_solver_config(problem,config);
assert(config.beta>0 && config.beta<=1,'run_cb2o:InvalidBeta', ...
    'CB2O beta must lie in (0,1].');
assert(config.epsilon_stop>=0,'run_cb2o:InvalidStoppingTolerance', ...
    'CB2O epsilon_stop must be nonnegative.');
diffusion_type = char(config.diffusion_type);
assert(any(strcmp(diffusion_type,{'anisotropic','isotropic'})), ...
    'run_cb2o:InvalidDiffusion', ...
    'CB2O diffusion_type must be anisotropic or isotropic.');

rng(config.seed,'twister');
V = initialize_particles(problem,config);
initial_ensemble = V;
trajectory = trajectory_initialize(problem,config);
trajectory.quantile_count = NaN(1,config.max_steps+1);
trajectory.quantile_threshold = NaN(1,config.max_steps+1);
[trajectory,~] = trajectory_record_cb2o(trajectory,0,V,problem,config);

iterations = 0;
exit_reason = 'max_steps';
stop_concentration = Inf;
timer = tic;
while iterations<config.max_steps && stop_concentration>config.epsilon_stop
    [consensus,~,~] = compute_cb2o_consensus( ...
        V,problem,config.alpha,config.beta);
    V_next = cb2o_step(V,consensus,config,randn(size(V)));
    % The CB2O stopping criterion uses the consensus from before the update.
    stop_concentration = sum((V_next-consensus).^2,'all') ...
        /(problem.dimension*config.particles);
    V = V_next;
    iterations = iterations+1;
    [trajectory,~] = trajectory_record_cb2o( ...
        trajectory,iterations,V,problem,config,stop_concentration);
end
if stop_concentration<=config.epsilon_stop
    exit_reason = 'concentration';
end
elapsed = toc(timer);

% Recompute the quantile selection and weights for the final ensemble.
[v_out,weights,final_diagnostics] = compute_cb2o_consensus( ...
    V,problem,config.alpha,config.beta);
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
result.final_cb2o_consensus = v_out;
result.output_source = 'fresh-final-cb2o-consensus';
result.trajectory = trajectory;
result.active_mask = trajectory.active_mask;
result.config = config;
result.problem_name = problem.name;
result.method = 'cb2o';
result.beta = config.beta;
result.quantile_count = final_diagnostics.quantile_count;
result.quantile_threshold = final_diagnostics.quantile_threshold;
result.quantile_selected_mask = final_diagnostics.selected_mask;
result.diffusion_type = diffusion_type;
result.stop_concentration = stop_concentration;
active_counts = trajectory.quantile_count(trajectory.active_mask);
result.mean_quantile_count = mean(active_counts);
result.min_quantile_count = min(active_counts);
result.max_quantile_count = max(active_counts);
result.final_quantile_count = final_diagnostics.quantile_count;
result.all_finite = all(isfinite(V),'all') && all(isfinite(v_out));
end
