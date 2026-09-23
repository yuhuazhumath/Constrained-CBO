function experiment = run_sensitivity(config,options)
%RUN_SENSITIVITY Run/checkpoint selected Algorithm-2 grid cells.
% options fields: repetitions, alpha_values, epsilon_values, pairs,
% output_file, resume, and store_full_results. pairs is an n-by-2
% [alpha,epsilon] list. Full solver execution is identical in both storage
% modes; compact mode discards heavy fields only after augmentation.
if nargin<1 || isempty(config)
    config = config_sensitivity();
end
if nargin<2 || isempty(options)
    options = struct();
end
assert(strcmp(config.family,'sensitivity-alpha-epsilon'), ...
    'run_sensitivity:WrongFamily', ...
    'Expected a sensitivity-alpha-epsilon configuration.');

repetitions = option_value(options,'repetitions',config.repetitions);
validateattributes(repetitions,{'double'},{'scalar','integer','positive', ...
    '<=',config.repetitions});
alpha_subset = option_value(options,'alpha_values',config.alpha_values);
epsilon_subset = option_value(options,'epsilon_values',config.epsilon_values);
validate_subset(alpha_subset,config.alpha_values,'alpha');
validate_subset(epsilon_subset,config.epsilon_values,'epsilon');
selected_mask = ismember(config.alpha_values,alpha_subset(:).')' ...
    & ismember(config.epsilon_values,epsilon_subset(:).');
if isfield(options,'pairs') && ~isempty(options.pairs)
    pairs = options.pairs;
    assert(size(pairs,2)==2,'run_sensitivity:InvalidPairs', ...
        'pairs must be an n-by-2 [alpha,epsilon] array.');
    pair_mask = false(size(selected_mask));
    for pair_index=1:size(pairs,1)
        alpha_index = find(config.alpha_values==pairs(pair_index,1),1);
        epsilon_index = find(config.epsilon_values==pairs(pair_index,2),1);
        assert(~isempty(alpha_index) && ~isempty(epsilon_index), ...
            'run_sensitivity:PairOutsideGrid', ...
            'Every requested pair must belong to the intended grid.');
        pair_mask(alpha_index,epsilon_index) = true;
    end
    selected_mask = selected_mask & pair_mask;
end
assert(any(selected_mask,'all'), ...
    'run_sensitivity:EmptySelection','No grid cells selected.');
output_file = char(option_value(options,'output_file',config.output_file));
resume = logical(option_value(options,'resume',true));
assert(isscalar(resume),'run_sensitivity:InvalidResume', ...
    'resume must be scalar.');
store_full_results = logical(option_value(options,'store_full_results',false));
assert(isscalar(store_full_results), ...
    'run_sensitivity:InvalidStorageOption', ...
    'store_full_results must be scalar.');
storage_mode = storage_mode_name(store_full_results);

signature = struct('family',config.family,'case_id',config.case_id, ...
    'benchmark_id',config.benchmark_id,'dimension',config.dimension, ...
    'problem_name',config.problem.name,'vstar',config.problem.vstar, ...
    'objective_star',config.problem.objective_star, ...
    'master_seed',config.master_seed,'repetitions',repetitions, ...
    'alpha_values',config.alpha_values,'epsilon_values',config.epsilon_values, ...
    'selected_mask',selected_mask,'resolved_cells',strip_cell_configs(config.cells), ...
    'base_solver_config',revision_solver_config_metadata( ...
    config.baseline_solver_config),'storage_mode',storage_mode);
if resume && ~isempty(output_file) && isfile(output_file)
    loaded = load(output_file,'experiment');
    experiment = loaded.experiment;
    assert(isfield(experiment,'signature') ...
        && isequaln(experiment.signature,signature), ...
        'run_sensitivity:CheckpointMismatch', ...
        'Existing output belongs to a different run selection or configuration.');
else
    experiment = initialize_sensitivity_experiment( ...
        config,signature,selected_mask,repetitions,output_file, ...
        store_full_results,storage_mode);
    checkpoint(output_file,experiment);
end

base_config = validate_solver_config(config.problem,config.baseline_solver_config);
for repetition=1:repetitions
    initial_seed = derive_repetition_seed(config.master_seed,repetition,0);
    solver_seed = derive_repetition_seed(config.master_seed,repetition,1);
    assert(experiment.initial_seeds(repetition)==initial_seed ...
        && experiment.solver_seeds(repetition)==solver_seed, ...
        'run_sensitivity:SeedMismatch', ...
        'Checkpoint seed metadata does not match deterministic derivation.');
    rng(initial_seed,'twister');
    base_config.initial_particles = [];
    base = initialize_particles(config.problem,base_config);
    if store_full_results
        if isempty(experiment.base_initializations{repetition})
            experiment.base_initializations{repetition} = base;
        else
            assert(isequal(experiment.base_initializations{repetition},base), ...
                'run_sensitivity:InitializationMismatch', ...
                'Saved initialization does not match deterministic seed derivation.');
        end
    end
    for alpha_index=1:numel(config.alpha_values)
        for epsilon_index=1:numel(config.epsilon_values)
            if ~selected_mask(alpha_index,epsilon_index) ...
                    || experiment.completed_mask(alpha_index,epsilon_index,repetition)
                continue
            end
            cell_config = config.cells(alpha_index,epsilon_index).solver_config;
            cell_config.seed = solver_seed;
            cell_config.initial_particles = base;
            result = run_constrained_cbo_independent_noise( ...
                config.problem,cell_config);
            result = augment_revision_result(result,config.problem, ...
                initial_seed,solver_seed,repetition);
            result.alpha_index = alpha_index;
            result.epsilon_index = epsilon_index;
            result.benchmark_id = config.benchmark_id;
            result.case_id = config.case_id;
            if ~store_full_results
                result = compact_revision_result(result);
            end
            experiment.results{alpha_index,epsilon_index,repetition} = result;
            experiment.completed_mask(alpha_index,epsilon_index,repetition) = true;
            experiment.completed_task_count = nnz(experiment.completed_mask);
            checkpoint(output_file,experiment);
        end
    end
end
required = repmat(selected_mask,1,1,repetitions);
experiment.is_complete = all(experiment.completed_mask(required));
checkpoint(output_file,experiment);
end

function experiment = initialize_sensitivity_experiment(config,signature,selected_mask,repetitions,output_file,store_full_results,storage_mode)
experiment = struct();
experiment.id = config.id;
experiment.family = config.family;
experiment.signature = signature;
experiment.case_id = config.case_id;
experiment.benchmark_id = config.benchmark_id;
experiment.dimension = config.dimension;
experiment.problem_name = config.problem.name;
experiment.problem_metadata = config.problem.metadata;
experiment.vstar = config.problem.vstar;
experiment.objective_star = config.problem.objective_star;
experiment.alpha_values = config.alpha_values;
experiment.epsilon_values = config.epsilon_values;
experiment.alpha_over_epsilon = config.alpha_over_epsilon;
experiment.axis_semantics = config.axis_semantics;
experiment.selected_mask = selected_mask;
experiment.results = cell(numel(config.alpha_values), ...
    numel(config.epsilon_values),repetitions);
experiment.completed_mask = false(size(experiment.results));
experiment.storage_mode = storage_mode;
experiment.initial_seeds = arrayfun(@(r)derive_repetition_seed( ...
    config.master_seed,r,0),1:repetitions);
experiment.solver_seeds = arrayfun(@(r)derive_repetition_seed( ...
    config.master_seed,r,1),1:repetitions);
experiment.base_solver_config = revision_solver_config_metadata( ...
    config.baseline_solver_config);
experiment.resolved_cells = strip_cell_configs(config.cells);
if store_full_results
    experiment.base_initializations = cell(1,repetitions);
    experiment.resolved_solver_configs = reshape({config.cells.solver_config}, ...
        size(config.cells));
    experiment.full_intended_config = config;
end
experiment.completed_task_count = 0;
experiment.is_complete = false;
experiment.master_seed = config.master_seed;
experiment.repetitions = repetitions;
experiment.output_file = output_file;
experiment.environment = get_environment_metadata();
experiment.created_at = char(datetime('now','TimeZone','UTC', ...
    'Format','yyyy-MM-dd''T''HH:mm:ssXXX'));
end

function cells = strip_cell_configs(cells)
cells = rmfield(cells,'solver_config');
end

function name = storage_mode_name(store_full_results)
if store_full_results
    name = 'full';
else
    name = 'compact';
end
end

function validate_subset(values,intended,name)
assert(isnumeric(values) && isvector(values) && ~isempty(values) ...
    && all(ismember(values,intended)), ...
    'run_sensitivity:SubsetOutsideGrid', ...
    'Requested %s values must be a nonempty subset of the intended axis.',name);
end

function value = option_value(options,name,default)
if isfield(options,name)
    value = options.(name);
else
    value = default;
end
end

function checkpoint(output_file,experiment)
if isempty(output_file)
    return
end
parent = fileparts(output_file);
if ~isempty(parent) && ~isfolder(parent)
    mkdir(parent);
end
save_revision_checkpoint(output_file,experiment);
end
