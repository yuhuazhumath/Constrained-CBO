function configs = config_thomson(master_seed)
%CONFIG_THOMSON Symmetry-reduced Thomson configurations from the manuscript.
arguments
    master_seed (1,1) double {mustBeInteger,mustBeNonnegative} = 47000
end
k_values = [2,3,8,15,56,470];
template = struct('id','','problem',struct(),'methods',{{}}, ...
    'solver_config',struct(),'repetitions',0,'master_seed',0,'output_file','');
configs = repmat(template,1,numel(k_values));
for i=1:numel(k_values)
    k = k_values(i);
    solver = default_solver_config();
    solver.particles = 50;
    solver.alpha = 50;
    solver.epsilon = 0.01;
    solver.lambda = 1;
    solver.sigma = 1;
    solver.gamma = 0.1;
    solver.max_steps = 2000;
    solver.concentration_tol = 1e-14;
    solver.improvement_tol = 0.01;
    solver.sigma_indep = 0.3;
    solver.store_trajectory = false;
    solver.initialization = struct('type','thomson_angular');
    solver.success_relative_objective_tol = 0.05;
    solver.success_l1_feasibility_tol = 1e-3;
    id = sprintf('figure7_thomson_k%d',k);
    configs(i) = struct('id',id,'problem',thomson_problem(k), ...
        'methods',{{'proposed-independent-noise'}},'solver_config',solver, ...
        'repetitions',100,'master_seed',master_seed+100*(i-1), ...
        'output_file',fullfile('results','raw','fig7', ...
        sprintf('%s_seed%d.mat',id,master_seed+100*(i-1))));
end
end
