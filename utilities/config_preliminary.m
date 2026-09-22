function configs = config_preliminary(master_seed)
%CONFIG_PRELIMINARY Reproducible configurations for preliminary comparisons.
arguments
    master_seed (1,1) double {mustBeInteger,mustBeNonnegative} = 12000
end
solver = default_solver_config();
solver.particles = 50;
solver.alpha = 30;
solver.epsilon = 0.01;
% CB2O uses beta=1/20 for the same-particle-count comparison.
solver.beta = 1/20;
solver.epsilon_stop = 0;
solver.diffusion_type = 'anisotropic';
solver.lambda = 1;
solver.sigma = 1;
solver.gamma = 0.01;
solver.max_steps = 300;
solver.concentration_tol = 0;
solver.initialization = struct('type','uniform_box','lower',-3,'upper',3);
solver.success_infinity_distance_tol = 0.01;

configs = repmat(base_config('',[],{},solver,100,master_seed),1,3);
configs(1) = base_config('figure2a_circle_equal',make_comparison_problem(1), ...
    {'proposed','projected-cbo','quadratic-penalty-cbo','cb2o'},solver,100,master_seed);
configs(2) = base_config('figure2b_circle_different',make_comparison_problem(2), ...
    {'proposed','projected-cbo','quadratic-penalty-cbo','cb2o'},solver,100,master_seed+100);
configs(3) = base_config('figure2c_parabola',make_comparison_problem(3), ...
    {'proposed','quadratic-penalty-cbo','cb2o'},solver,100,master_seed+200);
end

function problem = make_comparison_problem(case_id)
if case_id==1
    vhat = [1;-1]/sqrt(2);
    vstar = vhat;
else
    vhat = [0.5;1/3];
    if case_id==2
        vstar = [0.7817183879;0.6236315916];
    else
        vstar = [0.5427013393;0.2945247437];
    end
end
problem = ackley_problem(2,vhat,20,0.2,3);
if case_id<=2
    g = @(V) sum(V.^2,1)-1;
    gradg = @(V) 2*V;
    hessg = @(V) repmat(2*eye(2),1,1,size(V,2));
    problem = scalar_constraint(problem,'circle',g,gradg,hessg,vstar);
else
    g = @(V) V(1,:).^2-V(2,:);
    gradg = @(V) [2*V(1,:);-ones(1,size(V,2))];
    hessg = @(V) repmat(diag([2,0]),1,1,size(V,2));
    problem = scalar_constraint(problem,'parabola',g,gradg,hessg,vstar);
end
end

function config = base_config(id,problem,methods,solver,repetitions,master_seed)
config = struct('id',id,'problem',problem,'methods',{methods}, ...
    'solver_config',solver,'repetitions',repetitions,'master_seed',master_seed, ...
    'output_file',fullfile('results','raw','fig1_2', ...
    sprintf('%s_seed%d.mat',id,master_seed)));
end
