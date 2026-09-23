function config = config_sensitivity(case_id,master_seed)
%CONFIG_SENSITIVITY Grid of alpha and epsilon values.
% Tables 4 and 7 use case_id=3 and master_seed=36500.
if nargin<1 || isempty(case_id)
    case_id = 4;
end
benchmark = get_revision_benchmark_config(case_id);
if nargin<2 || isempty(master_seed)
    master_seed = benchmark.master_seed;
end
validateattributes(master_seed,{'double'},{'scalar','integer','nonnegative'});

alpha_values = [10,20,30,50,80];
epsilon_values = [0.005,0.01,0.02,0.05,0.1];
cell_template = struct('alpha_index',0,'epsilon_index',0,'alpha',NaN, ...
    'epsilon',NaN,'alpha_over_epsilon',NaN,'solver_config',struct());
cells = repmat(cell_template,numel(alpha_values),numel(epsilon_values));
for alpha_index=1:numel(alpha_values)
    for epsilon_index=1:numel(epsilon_values)
        solver = benchmark.solver_config;
        solver.alpha = alpha_values(alpha_index);
        solver.epsilon = epsilon_values(epsilon_index);
        cells(alpha_index,epsilon_index) = struct( ...
            'alpha_index',alpha_index,'epsilon_index',epsilon_index, ...
            'alpha',solver.alpha,'epsilon',solver.epsilon, ...
            'alpha_over_epsilon',solver.alpha/solver.epsilon, ...
            'solver_config',solver);
    end
end

config = struct();
config.id = sprintf('revision_sensitivity_alpha_epsilon_d20_case%d',case_id);
config.family = 'sensitivity-alpha-epsilon';
config.case_id = case_id;
config.dimension = benchmark.problem.dimension;
config.benchmark_id = benchmark.id;
config.problem = benchmark.problem;
config.method = 'proposed-independent-noise';
config.baseline_solver_config = benchmark.solver_config;
config.baseline_experiment_config = benchmark;
config.alpha_values = alpha_values;
config.epsilon_values = epsilon_values;
config.alpha_over_epsilon = alpha_values(:)./epsilon_values;
config.cells = cells;
config.repetitions = benchmark.repetitions;
config.master_seed = master_seed;
config.baseline_pair = struct('alpha',benchmark.solver_config.alpha, ...
    'epsilon',benchmark.solver_config.epsilon);
config.axis_semantics = struct('row','alpha','column','epsilon');
config.output_file = fullfile('results','raw','revision', ...
    'sensitivity',sprintf('case%d_seed%d.mat', ...
    case_id,master_seed));
end
