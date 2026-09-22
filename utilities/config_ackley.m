function configs = config_ackley(master_seed)
%CONFIG_ACKLEY Seven manuscript Ackley panels in canonical panel order.
arguments
    master_seed (1,1) double {mustBeInteger,mustBeNonnegative} = 36000
end
base = default_solver_config();
base.particles = 100;
base.alpha = 50;
base.epsilon = 0.01;
base.lambda = 1;
base.sigma = 1;
base.gamma = 0.1;
base.initialization = struct('type','uniform_box','lower',-3,'upper',3);
base.success_infinity_distance_tol = 0.1;

problems = {ackley_segment(),ackley_sphere(3), ...
    ackley_paraboloid(3),ackley_planes(), ...
    ackley_ball(),ackley_sphere(20), ...
    ackley_paraboloid(20)};
ids = {'figure6a_d3_case1','figure6b_d3_case3','figure6c_d3_case4', ...
    'figure6d_d3_case5','figure6e_d20_case2','figure6f_d20_case3', ...
    'figure6g_d20_case4'};
template = struct('id','','problem',struct(),'methods',{{}}, ...
    'solver_config',struct(),'method_overrides',struct([]), ...
    'proposed_solver','','repetitions',0,'master_seed',0,'output_file','');
configs = repmat(template,1,7);
methods = {'proposed','quadratic-penalty-cbo','cb2o'};
for i=1:7
    solver = base;
    if i<=4
        solver.max_steps = 1000;
        solver.concentration_tol = 1e-14;
        proposed_solver = 'algorithm1';
    else
        solver.max_steps = 5000;
        solver.concentration_tol = 1e-5;
        proposed_solver = 'algorithm2';
        if i==5
            solver.improvement_tol = 0.05;
            solver.sigma_indep = 0.3;
        elseif i==6
            solver.improvement_tol = 0.01;
            solver.sigma_indep = 0.3;
        else
            solver.improvement_tol = 0.001;
            solver.sigma_indep = 1;
        end
    end
    overrides = comparison_method_overrides(solver.epsilon,solver.beta, ...
        solver.diffusion_type,false);
    configs(i) = struct('id',ids{i},'problem',problems{i}, ...
        'methods',{methods},'solver_config',solver, ...
        'method_overrides',overrides,'proposed_solver',proposed_solver, ...
        'repetitions',100, ...
        'master_seed',master_seed+100*(i-1), ...
        'output_file',fullfile('results','raw','fig6', ...
        sprintf('%s_seed%d.mat',ids{i},master_seed+100*(i-1))));
end
end
