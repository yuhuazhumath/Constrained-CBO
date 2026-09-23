function configs = config_simple(master_seed)
%CONFIG_SIMPLE Simple problems with exact snapshot indices 0,5,50,100.
arguments
    master_seed (1,1) double {mustBeInteger,mustBeNonnegative} = 24000
end
solver = default_solver_config();
solver.particles = 50;
solver.alpha = 50;
solver.epsilon = 0.01;
solver.lambda = 1;
solver.sigma = 1;
solver.gamma = 0.1;
solver.max_steps = 100;
solver.concentration_tol = 1e-14;
solver.initialization = struct('type','uniform_box','lower',-3,'upper',3);
solver.snapshot_steps = [0,5,50,100];
solver.success_infinity_distance_tol = 0.1;
problems = {quadratic_segment(),quadratic_ellipse(), ...
    quadratic_line()};
ids = {'figure4_5_segment','figure4_5_ellipse','figure4_5_line'};
template = struct('id','','problem',struct(),'methods',{{}}, ...
    'solver_config',struct(),'method_overrides',struct([]), ...
    'proposed_solver','','repetitions',0,'master_seed',0,'output_file','');
configs = repmat(template,1,3);
methods = {'proposed','quadratic-penalty-cbo','cb2o'};
overrides = comparison_method_overrides(solver.epsilon,solver.beta, ...
    solver.diffusion_type,true);
for i=1:3
    configs(i) = struct('id',ids{i},'problem',problems{i}, ...
        'methods',{methods},'solver_config',solver, ...
        'method_overrides',overrides,'proposed_solver','algorithm1', ...
        'repetitions',100, ...
        'master_seed',master_seed+100*(i-1), ...
        'output_file',fullfile('results','raw','fig4_5', ...
        sprintf('%s_seed%d.mat',ids{i},master_seed+100*(i-1))));
end
end
