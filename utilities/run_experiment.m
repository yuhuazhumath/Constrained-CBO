function experiment = run_experiment(config)
%RUN_EXPERIMENT Reproducible multi-method runner; saves raw data only.
arguments
    config struct
end
required = {'id','problem','methods','solver_config','repetitions','master_seed'};
for i=1:numel(required)
    assert(isfield(config,required{i}),'run_experiment:MissingField', ...
        'Configuration is missing %s.',required{i});
end
problem = config.problem;
method_count = numel(config.methods);
signature = experiment_checkpoint_signature(config);
checkpoint_requested = isfield(config,'output_file') ...
    && ~isempty(config.output_file);
if checkpoint_requested
    output_file = char(config.output_file);
    parent = fileparts(output_file);
    if ~isempty(parent) && ~isfolder(parent)
        mkdir(parent);
    end
    if isfile(output_file)
        loaded = load(output_file,'experiment');
        experiment = loaded.experiment;
        assert(isfield(experiment,'signature') ...
            && isequaln(experiment.signature,signature), ...
            'run_experiment:CheckpointMismatch', ...
            ['Existing output belongs to a different case, method set, ', ...
            'parameter, reference, seed, or storage configuration.']);
        validate_checkpoint(experiment,config,method_count);
        experiment.resumed_from_checkpoint = true;
    else
        experiment = initialize_experiment( ...
            config,problem,method_count,signature);
        experiment.output_file = output_file;
        experiment = checkpoint_experiment(output_file,experiment);
    end
else
    output_file = '';
    experiment = initialize_experiment( ...
        config,problem,method_count,signature);
end
base_config = validate_solver_config(problem,config.solver_config);
for repetition=1:config.repetitions
    if all(experiment.completed_mask(:,repetition))
        continue
    end
    initial_seed = derive_repetition_seed(config.master_seed,repetition,0);
    solver_seed = derive_repetition_seed(config.master_seed,repetition,1);
    assert(experiment.initial_seeds(repetition)==initial_seed ...
        && experiment.solver_seeds(repetition)==solver_seed, ...
        'run_experiment:SeedMismatch', ...
        'Checkpoint seed metadata does not match deterministic derivation.');
    rng(initial_seed,'twister');
    base_config.initial_particles = [];
    regenerated_base = initialize_particles(problem,base_config);
    if isempty(experiment.base_initializations{repetition})
        experiment.base_initializations{repetition} = regenerated_base;
    else
        assert(isequal(experiment.base_initializations{repetition}, ...
            regenerated_base),'run_experiment:InitializationMismatch', ...
            'Saved initialization does not match deterministic seed derivation.');
    end
    base = experiment.base_initializations{repetition};
    try
        for method_index=1:method_count
            if experiment.completed_mask(method_index,repetition)
                continue
            end
            method = config.methods{method_index};
            solver_config = resolve_method_solver_config(config,method);
            solver_config.seed = solver_seed;
            solver_config.initial_particles = base;
            result = dispatch_solver(method,problem,solver_config,config);
            result.initial_seed = initial_seed;
            result.stochastic_seed = solver_seed;
            result.solver_seed = solver_seed;
            result.repetition = repetition;
            result.case_id = config.id;
            result.method = method;
            result.initialization_id = sprintf('%s-repetition-%d-seed-%d', ...
                config.id,repetition,initial_seed);
            if isfield(result,'objective_error')
                result.objective_gap = result.objective_error;
            else
                result.objective_gap = NaN;
            end
            result.all_finite = result_is_finite(result);
            experiment.results{method_index,repetition} = result;
            experiment.completed_mask(method_index,repetition) = true;
        end
    catch exception
        experiment = update_completion(experiment);
        if checkpoint_requested
            checkpoint_experiment(output_file,experiment);
        end
        rethrow(exception)
    end
    experiment = update_completion(experiment);
    if checkpoint_requested
        experiment = checkpoint_experiment(output_file,experiment);
    end
end
experiment = update_completion(experiment);
end

function experiment = initialize_experiment(config,problem,method_count,signature)
experiment = struct();
experiment.id = config.id;
experiment.signature = signature;
experiment.storage_mode = signature.storage_mode;
experiment.problem_name = problem.name;
experiment.problem_metadata = problem.metadata;
experiment.vstar = problem.vstar;
experiment.objective_star = problem.objective_star;
experiment.methods = config.methods;
experiment.results = cell(method_count,config.repetitions);
experiment.base_initializations = cell(1,config.repetitions);
experiment.completed_mask = false(method_count,config.repetitions);
experiment.completed_task_count = 0;
experiment.completed_repetitions = 0;
experiment.is_complete = false;
experiment.master_seed = config.master_seed;
experiment.repetitions = config.repetitions;
experiment.initial_seeds = signature.initial_seeds;
experiment.solver_seeds = signature.solver_seeds;
experiment.solver_config = config.solver_config;
if isfield(config,'method_overrides')
    experiment.method_overrides = config.method_overrides;
else
    experiment.method_overrides = struct([]);
end
if isfield(config,'proposed_solver')
    experiment.proposed_solver = config.proposed_solver;
else
    experiment.proposed_solver = '';
end
experiment.output_file = '';
experiment.checkpoint_write_count = 0;
experiment.resumed_from_checkpoint = false;
experiment.environment = get_environment_metadata();
experiment.created_at = char(datetime('now','TimeZone','UTC', ...
    'Format','yyyy-MM-dd''T''HH:mm:ssXXX'));
end

function validate_checkpoint(experiment,config,method_count)
assert(isequal(size(experiment.results), ...
    [method_count,config.repetitions]) ...
    && isequal(size(experiment.completed_mask), ...
    [method_count,config.repetitions]) ...
    && numel(experiment.base_initializations)==config.repetitions, ...
    'run_experiment:InvalidCheckpointShape', ...
    'Existing checkpoint arrays do not match the requested experiment.');
completed_indices = find(experiment.completed_mask);
for j=1:numel(completed_indices)
    index = completed_indices(j);
    assert(~isempty(experiment.results{index}), ...
        'run_experiment:InvalidCheckpointCompletion', ...
        'A completed checkpoint task has no stored result.');
end
end

function experiment = update_completion(experiment)
experiment.completed_task_count = nnz(experiment.completed_mask);
experiment.completed_repetitions = sum(all(experiment.completed_mask,1));
experiment.is_complete = all(experiment.completed_mask,'all');
end

function experiment = checkpoint_experiment(output_file,experiment)
experiment.checkpoint_write_count = experiment.checkpoint_write_count+1;
save_revision_checkpoint(output_file,experiment);
end

function result = dispatch_solver(method,problem,config,experiment_config)
switch method
    case 'proposed'
        proposed_solver = 'algorithm1';
        if isfield(experiment_config,'proposed_solver') ...
                && ~isempty(experiment_config.proposed_solver)
            proposed_solver = experiment_config.proposed_solver;
        end
        if strcmp(proposed_solver,'algorithm1')
            result = run_constrained_cbo(problem,config);
        elseif strcmp(proposed_solver,'algorithm2')
            result = run_constrained_cbo_independent_noise(problem,config);
        else
            error('run_experiment:UnknownProposedSolver', ...
                'Unknown proposed solver "%s".',proposed_solver);
        end
    case 'proposed-independent-noise'
        result = run_constrained_cbo_independent_noise(problem,config);
    case 'projected-cbo'
        result = run_projected_cbo(problem,config);
    case 'quadratic-penalty-cbo'
        result = run_quadratic_penalty_cbo(problem,config);
    case 'cb2o'
        result = run_cb2o(problem,config);
    otherwise
        error('run_experiment:UnknownMethod','Unknown method "%s".',method);
end
end

function finite = result_is_finite(result)
scalars = [result.objective_value,result.G_value, ...
    result.feasibility_error,result.iterations,result.runtime];
if isfield(result,'objective_gap') && ~isnan(result.objective_gap)
    scalars(end+1) = result.objective_gap;
end
if isfield(result,'distance_to_vstar') ...
        && ~isnan(result.distance_to_vstar)
    scalars(end+1) = result.distance_to_vstar;
end
finite = all(isfinite(result.v_out),'all') && all(isfinite(scalars));
if isfield(result,'final_ensemble')
    finite = finite && all(isfinite(result.final_ensemble),'all');
end
end
