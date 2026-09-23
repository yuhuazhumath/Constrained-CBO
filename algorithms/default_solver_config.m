function config = default_solver_config()
%DEFAULT_SOLVER_CONFIG Default parameters shared by the solvers.
config = struct();
config.particles = 50;
config.alpha = 50;
config.epsilon = 0.01;
config.beta = 1/20;
config.epsilon_stop = 0;
config.diffusion_type = 'anisotropic';
config.lambda = 1;
config.sigma = 1;
config.gamma = 0.1;
config.max_steps = 100;
config.concentration_tol = 1e-14;
config.improvement_tol = 0.01;
config.sigma_indep = 0.3;
config.seed = 1;
config.initial_particles = [];
config.initialization = struct('type','uniform_box','lower',-3,'upper',3);
config.store_trajectory = true;
config.snapshot_steps = [];
config.success_distance_tol = [];
config.success_infinity_distance_tol = [];
config.success_feasibility_tol = [];
config.success_relative_objective_tol = [];
config.success_l1_feasibility_tol = [];
end
