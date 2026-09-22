function diagnostic = run_conditioning_k470_diagnostic()
%RUN_CONDITIONING_K470_DIAGNOSTIC Condition numbers for k=470, repetitions 1:3.
% Instrument Algorithm 2 using the same constrained_cbo_step updates.

diagnostic_code_dir = fileparts(mfilename('fullpath'));
root = fileparts(diagnostic_code_dir);
addpath(fullfile(root,'algorithms'),fullfile(root,'examples'), ...
    fullfile(root,'utilities'),fullfile(root,'problems'));
allcfg = config_thomson();
matches = arrayfun(@(c)c.problem.metadata.k==470,allcfg);
assert(nnz(matches)==1,'Expected exactly one canonical k=470 case.');
cfg = allcfg(matches);
problem = cfg.problem;
base_config = validate_solver_config(problem,cfg.solver_config);
signature = experiment_checkpoint_signature(cfg);

assert(problem.metadata.k==470);
assert(problem.num_free_electrons==469);
assert(problem.dimension==1407);
assert(problem.metadata.ambient_physical_dimension==1410);
assert(cfg.master_seed==47500 && cfg.repetitions==100);
assert(base_config.particles==50);
q = base_config.gamma/base_config.epsilon;

production = load(fullfile(root,cfg.output_file),'experiment');
production = production.experiment;
assert(production.is_complete && production.completed_repetitions==100);
assert(isequaln(production.signature,signature));

repetition_ids = 1:3;
run_diagnostics = cell(1,3);
final_outputs = cell(1,3);
reproduction = [];
formula_validation = struct();
total_timer = tic;

for run_index = 1:3
    repetition = repetition_ids(run_index);
    initial_seed = signature.initial_seeds(repetition);
    solver_seed = signature.solver_seeds(repetition);
    assert(initial_seed==derive_repetition_seed(cfg.master_seed,repetition,0));
    assert(solver_seed==derive_repetition_seed(cfg.master_seed,repetition,1));

    rng(initial_seed,'twister');
    base_config.initial_particles = [];
    initial_particles = initialize_particles(problem,base_config);
    assert(isequal(initial_particles,production.base_initializations{repetition}), ...
        'Canonical initialization does not match production checkpoint.');

    solver_config = resolve_method_solver_config(cfg,cfg.methods{1});
    solver_config.seed = solver_seed;
    solver_config.initial_particles = initial_particles;
    [result,run_diag,validation] = run_instrumented_algorithm2( ...
        problem,solver_config,repetition,run_index==1,q);
    if run_index==1
        formula_validation = validation;
    end

    result.initial_seed = initial_seed;
    result.stochastic_seed = solver_seed;
    result.solver_seed = solver_seed;
    result.repetition = repetition;
    result.case_id = cfg.id;
    result.method = cfg.methods{1};
    result.initialization_id = sprintf('%s-repetition-%d-seed-%d', ...
        cfg.id,repetition,initial_seed);
    result.objective_gap = result.objective_error;

    production_result = production.results{1,repetition};
    comparison = compare_results(result,production_result,repetition);
    if isempty(reproduction)
        reproduction = repmat(comparison,1,3);
    end
    reproduction(run_index) = comparison; %#ok<AGROW>
    assert(reproduction(run_index).pass, ...
        'Diagnostic repetition %d did not exactly reproduce production.',repetition);

    run_diag.summary = distribution_summary(run_diag);
    run_diagnostics{run_index} = run_diag;
    final_outputs{run_index} = compact_final_output(result);
end

diagnostic = struct();
diagnostic.schema_version = 1;
diagnostic.description = ['Conditioning diagnostic for the exact pre-update ', ...
    'Thomson implicit matrices in canonical k=470 repetitions 1--3.'];
diagnostic.repository_branch = '';
diagnostic.repository_head = '';
diagnostic.matlab_version = version;
diagnostic.matlab_release = version('-release');
diagnostic.created_at = char(datetime('now','TimeZone','UTC', ...
    'Format','yyyy-MM-dd''T''HH:mm:ssXXX'));
diagnostic.case_id = cfg.id;
diagnostic.production_file = cfg.output_file;
diagnostic.repetition_ids = repetition_ids;
diagnostic.initialization_seeds = signature.initial_seeds(repetition_ids);
diagnostic.solver_seeds = signature.solver_seeds(repetition_ids);
diagnostic.problem_metadata = problem.metadata;
diagnostic.solver_config = revision_solver_config_metadata(cfg.solver_config);
diagnostic.master_seed = cfg.master_seed;
diagnostic.canonical_repetitions = cfg.repetitions;
diagnostic.q = q;
diagnostic.formulas = struct( ...
    'hessian','8*x*x'' + 4*(r^2-1)*I3', ...
    'lambda_tangential','1 + 4*q*(r^2-1), multiplicity 2', ...
    'lambda_radial','1 + q*(12*r^2-4)', ...
    'singular_values','abs([lambda_tangential lambda_tangential lambda_radial])', ...
    'full_condition','max of all block singular values / min of all block singular values');
diagnostic.reference = reference_values(q);
diagnostic.formula_validation = formula_validation;
diagnostic.run_diagnostics = run_diagnostics;
diagnostic.final_outputs = final_outputs;
diagnostic.reproduction = reproduction;
diagnostic.combined_summary = combined_summary(run_diagnostics);
diagnostic.combined_block_statistics = combine_block_statistics(run_diagnostics);
diagnostic.total_diagnostic_runtime_seconds = toc(total_timer);
diagnostic.structured_solve_timing_collected = false;
diagnostic.structured_solve_timing_note = ['Not collected: constrained_cbo_step ', ...
    'does not expose a solve-only hook, and production code was not modified.'];
diagnostic.percentile_definition = ['Linear interpolation at index ', ...
    '1+(n-1)*p after sorting finite observed values.'];

outdir = fullfile(root,'results','diagnostics','conditioning');
if ~isfolder(outdir)
    mkdir(outdir);
end
output_file = fullfile(outdir,'k470_conditioning_reps1_3.mat');
assert(~isfile(output_file), ...
    'Refusing to overwrite the retained diagnostic: %s',output_file);
diagnostic.output_file = output_file;
save(output_file,'diagnostic','-v7.3');
fprintf('Saved %s\n',output_file);
for j=1:3
    fprintf('rep %d: %d iterations, exact reproduction=%d\n', ...
        repetition_ids(j),run_diagnostics{j}.iterations,reproduction(j).pass);
end
end

function [result,diag,validation] = run_instrumented_algorithm2(problem,config,repetition,do_validation,q)
config = validate_solver_config(problem,config);
rng(config.seed,'twister');
V = initialize_particles(problem,config);
initial_ensemble = V;
trajectory = trajectory_initialize(problem,config);
trajectory = trajectory_record(trajectory,0,V,problem,config);
best_objective = Inf;
best_consensus = [];
iterations = 0;
restarts = 0;
concentration_events = 0;
incumbent_updates = 0;
diag = initialize_diagnostics(config,problem,repetition);
validation = struct();
timer = tic;
while true
    concentration = trajectory.concentration(iterations+1);
    terminal_event = iterations>=config.max_steps;
    if concentration<=config.concentration_tol
        concentration_events = concentration_events+1;
        [current_consensus,~,~] = compute_consensus(V,problem.E,config.alpha);
        current_objective = problem.E(current_consensus);
        previous_best_objective = best_objective;
        [best_objective,best_consensus,should_stop] = ...
            update_algorithm2_incumbent(current_objective,current_consensus, ...
            best_objective,best_consensus,config.improvement_tol);
        incumbent_updates = incumbent_updates+double(best_objective<previous_best_objective);
        if terminal_event
            exit_reason = 'max_steps';
            break
        end
        if should_stop
            exit_reason = 'improvement';
            break
        end
        V = V+config.sigma_indep*sqrt(config.gamma)*randn(size(V));
        restarts = restarts+1;
        trajectory = trajectory_record(trajectory,iterations,V,problem,config);
    end
    if terminal_event
        exit_reason = 'max_steps';
        break
    end

    solve_iteration = iterations+1;
    rng_before_diagnostic = rng;
    [diag,blocks] = record_conditioning(diag,V,q,solve_iteration,repetition);
    if do_validation && solve_iteration==1
        validation = validate_formulas(problem,V,q,blocks);
    end
    assert(isequaln(rng_before_diagnostic,rng), ...
        'Diagnostic instrumentation changed the RNG state.');
    Z = randn(size(V));
    V = constrained_cbo_step(V,problem,config,Z);
    iterations = iterations+1;
    trajectory = trajectory_record(trajectory,iterations,V,problem,config);
end
elapsed = toc(timer);
if isempty(best_consensus)
    result = finalize_solver_result(V,problem,config,trajectory,iterations, ...
        exit_reason,elapsed,config.seed);
    result.output_source = 'final-consensus-fallback';
else
    result = finalize_solver_result(V,problem,config,trajectory,iterations, ...
        exit_reason,elapsed,config.seed,best_consensus);
end
result.initial_ensemble = initial_ensemble;
result.restarts = restarts;
result.restart_count = restarts;
result.concentration_event_count = concentration_events;
result.best_incumbent_update_count = incumbent_updates;
result.best_objective = best_objective;
result.best_consensus = best_consensus;
result.method = 'constrained-cbo-independent-noise';

diag.iterations = iterations;
diag.full_sigma_min = diag.full_sigma_min(1:iterations,:);
diag.full_sigma_max = diag.full_sigma_max(1:iterations,:);
diag.full_cond2 = diag.full_cond2(1:iterations,:);
diag.inverse_norm2 = diag.inverse_norm2(1:iterations,:);
diag.per_iteration = per_iteration_summary(diag);
diag.block_solves_per_iteration = problem.num_free_electrons*config.particles;
diag.block_solves_total = diag.block_solves_per_iteration*iterations;
diag.runtime_seconds = elapsed;
assert(diag.block_statistics.total_blocks==diag.block_solves_total);
end

function diag = initialize_diagnostics(config,problem,repetition)
shape = [config.max_steps,config.particles];
diag = struct();
diag.repetition = repetition;
diag.full_sigma_min = NaN(shape);
diag.full_sigma_max = NaN(shape);
diag.full_cond2 = NaN(shape);
diag.inverse_norm2 = NaN(shape);
diag.block_statistics = struct( ...
    'sigma_min_thresholds',[1e-1 1e-2 1e-3 1e-4 1e-6 1e-8], ...
    'sigma_min_counts',zeros(1,6), ...
    'cond2_thresholds',[1e2 1e3 1e4 1e6 1e8], ...
    'cond2_counts',zeros(1,5), ...
    'total_blocks',0, ...
    'global_min_sigma_min',Inf, ...
    'global_min_location',empty_block_location(), ...
    'global_max_block_cond2',-Inf, ...
    'global_max_location',empty_block_location());
diag.worst_full_system = empty_full_location();
diag.free_blocks_per_particle = problem.num_free_electrons;
diag.particles = config.particles;
end

function [diag,blocks] = record_conditioning(diag,V,q,iteration,repetition)
f = diag.free_blocks_per_particle;
n = diag.particles;
X = reshape(V,3,f,n);
r2 = reshape(sum(X.^2,1),f,n);
lambda_tan = 1+4*q*(r2-1);
lambda_rad = 1+q*(12*r2-4);
block_sigma_min = min(abs(lambda_tan),abs(lambda_rad));
block_sigma_max = max(abs(lambda_tan),abs(lambda_rad));
block_cond2 = block_sigma_max./block_sigma_min;
block_cond2(block_sigma_min==0) = Inf;
full_sigma_min = min(block_sigma_min,[],1);
full_sigma_max = max(block_sigma_max,[],1);
full_cond2 = full_sigma_max./full_sigma_min;
full_cond2(full_sigma_min==0) = Inf;

diag.full_sigma_min(iteration,:) = full_sigma_min;
diag.full_sigma_max(iteration,:) = full_sigma_max;
diag.full_cond2(iteration,:) = full_cond2;
diag.inverse_norm2(iteration,:) = 1./full_sigma_min;

stats = diag.block_statistics;
for j=1:numel(stats.sigma_min_thresholds)
    stats.sigma_min_counts(j) = stats.sigma_min_counts(j) ...
        +nnz(block_sigma_min<stats.sigma_min_thresholds(j));
end
for j=1:numel(stats.cond2_thresholds)
    stats.cond2_counts(j) = stats.cond2_counts(j) ...
        +nnz(block_cond2>stats.cond2_thresholds(j));
end
stats.total_blocks = stats.total_blocks+numel(block_sigma_min);

[minimum_sigma,linear_index] = min(block_sigma_min(:));
if minimum_sigma<stats.global_min_sigma_min
    [electron,particle] = ind2sub([f,n],linear_index);
    stats.global_min_sigma_min = minimum_sigma;
    stats.global_min_location = block_location(repetition,iteration,particle, ...
        electron,r2(electron,particle),lambda_tan(electron,particle), ...
        lambda_rad(electron,particle),block_sigma_min(electron,particle), ...
        block_sigma_max(electron,particle),block_cond2(electron,particle));
end
[maximum_cond,linear_index] = max(block_cond2(:));
if maximum_cond>stats.global_max_block_cond2
    [electron,particle] = ind2sub([f,n],linear_index);
    stats.global_max_block_cond2 = maximum_cond;
    stats.global_max_location = block_location(repetition,iteration,particle, ...
        electron,r2(electron,particle),lambda_tan(electron,particle), ...
        lambda_rad(electron,particle),block_sigma_min(electron,particle), ...
        block_sigma_max(electron,particle),block_cond2(electron,particle));
end
diag.block_statistics = stats;

[worst_cond,particle] = max(full_cond2);
if worst_cond>diag.worst_full_system.cond2
    [~,min_electron] = min(block_sigma_min(:,particle));
    [~,max_electron] = max(block_sigma_max(:,particle));
    diag.worst_full_system = struct( ...
        'repetition',repetition,'iteration',iteration,'particle',particle, ...
        'sigma_min',full_sigma_min(particle), ...
        'sigma_max',full_sigma_max(particle),'cond2',worst_cond, ...
        'min_singular_block',block_location(repetition,iteration,particle, ...
        min_electron,r2(min_electron,particle),lambda_tan(min_electron,particle), ...
        lambda_rad(min_electron,particle),block_sigma_min(min_electron,particle), ...
        block_sigma_max(min_electron,particle),block_cond2(min_electron,particle)), ...
        'max_singular_block',block_location(repetition,iteration,particle, ...
        max_electron,r2(max_electron,particle),lambda_tan(max_electron,particle), ...
        lambda_rad(max_electron,particle),block_sigma_min(max_electron,particle), ...
        block_sigma_max(max_electron,particle),block_cond2(max_electron,particle)));
end
blocks = struct('r2',r2,'lambda_tan',lambda_tan,'lambda_rad',lambda_rad, ...
    'sigma_min',block_sigma_min,'sigma_max',block_sigma_max,'cond2',block_cond2);
end

function validation = validate_formulas(problem,V,q,blocks)
pairs = [1 1; 25 235; 50 469];
max_matrix_error = 0;
max_singular_error = 0;
direct_blocks = cell(1,size(pairs,1));
analytic_singular_values = [];
for j=1:size(pairs,1)
    particle = pairs(j,1);
    electron = pairs(j,2);
    rows = (3*electron-2):(3*electron);
    H = problem.hessG(V(:,particle));
    production_A = eye(3)+q*full(H(rows,rows));
    x = V(rows,particle);
    formula_A = (1+4*q*(x.'*x-1))*eye(3)+8*q*(x*x.');
    max_matrix_error = max(max_matrix_error,norm(production_A-formula_A,2));
    direct = sort(svd(production_A),'descend');
    analytic = sort([abs(blocks.lambda_tan(electron,particle)); ...
        abs(blocks.lambda_tan(electron,particle)); ...
        abs(blocks.lambda_rad(electron,particle))],'descend');
    max_singular_error = max(max_singular_error,max(abs(direct-analytic)));
    direct_blocks{j} = production_A;
    analytic_singular_values = [analytic_singular_values; analytic]; %#ok<AGROW>
end
small_blockdiag = blkdiag(direct_blocks{:});
direct_subset_condition = cond(small_blockdiag,2);
analytic_subset_condition = max(analytic_singular_values)/min(analytic_singular_values);
subset_error = abs(direct_subset_condition-analytic_subset_condition);
scale = max(1,max(analytic_singular_values));
assert(max_matrix_error<=100*eps(scale));
assert(max_singular_error<=1000*eps(scale));
assert(subset_error<=1000*eps(max(1,direct_subset_condition)));
validation = struct('iteration',1,'particle_electron_pairs',pairs, ...
    'max_matrix_2norm_error',max_matrix_error, ...
    'max_singular_value_absolute_error',max_singular_error, ...
    'direct_subset_condition',direct_subset_condition, ...
    'analytic_subset_condition',analytic_subset_condition, ...
    'subset_condition_absolute_error',subset_error,'pass',true);
end

function summary = per_iteration_summary(diag)
K = diag.iterations;
summary = struct();
summary.minimum_full_sigma_min = min(diag.full_sigma_min,[],2);
summary.median_full_sigma_min = row_percentile(diag.full_sigma_min,50);
summary.median_full_cond2 = row_percentile(diag.full_cond2,50);
summary.percentile95_full_cond2 = row_percentile(diag.full_cond2,95);
summary.maximum_full_cond2 = max(diag.full_cond2,[],2);
summary.median_inverse_norm2 = row_percentile(diag.inverse_norm2,50);
summary.maximum_inverse_norm2 = max(diag.inverse_norm2,[],2);
assert(numel(summary.minimum_full_sigma_min)==K);
end

function values = row_percentile(matrix,p)
values = zeros(size(matrix,1),1);
for row=1:size(matrix,1)
    values(row) = percentile_linear(matrix(row,:),p);
end
end

function summary = distribution_summary(diag)
sigma = diag.full_sigma_min(:);
condition = diag.full_cond2(:);
inverse = diag.inverse_norm2(:);
summary = struct();
summary.system_count = numel(condition);
summary.full_sigma_min_percentiles = values_at_percentiles(sigma,[0 1 5 50 95 99 100]);
summary.full_cond2_percentiles = values_at_percentiles(condition,[0 50 90 95 99 100]);
summary.inverse_norm2_percentiles = values_at_percentiles(inverse,[50 95 99 100]);
summary.full_cond2_thresholds = [1e2 1e3 1e4 1e6];
summary.full_cond2_counts = arrayfun(@(t)nnz(condition>t),summary.full_cond2_thresholds);
summary.full_cond2_fractions = summary.full_cond2_counts/numel(condition);
summary.full_sigma_min_thresholds = [1e-1 1e-2 1e-3 1e-6];
summary.full_sigma_min_counts = arrayfun(@(t)nnz(sigma<t),summary.full_sigma_min_thresholds);
summary.full_sigma_min_fractions = summary.full_sigma_min_counts/numel(sigma);
end

function summary = combined_summary(runs)
sigma = [];
condition = [];
inverse = [];
for j=1:numel(runs)
    sigma = [sigma; runs{j}.full_sigma_min(:)]; %#ok<AGROW>
    condition = [condition; runs{j}.full_cond2(:)]; %#ok<AGROW>
    inverse = [inverse; runs{j}.inverse_norm2(:)]; %#ok<AGROW>
end
summary = struct();
summary.system_count = numel(condition);
summary.full_sigma_min_percentiles = values_at_percentiles(sigma,[0 1 5 50 95 99 100]);
summary.full_cond2_percentiles = values_at_percentiles(condition,[0 50 90 95 99 100]);
summary.inverse_norm2_percentiles = values_at_percentiles(inverse,[50 95 99 100]);
summary.full_cond2_thresholds = [1e2 1e3 1e4 1e6];
summary.full_cond2_counts = arrayfun(@(t)nnz(condition>t),summary.full_cond2_thresholds);
summary.full_cond2_fractions = summary.full_cond2_counts/numel(condition);
summary.full_sigma_min_thresholds = [1e-1 1e-2 1e-3 1e-6];
summary.full_sigma_min_counts = arrayfun(@(t)nnz(sigma<t),summary.full_sigma_min_thresholds);
summary.full_sigma_min_fractions = summary.full_sigma_min_counts/numel(sigma);
end

function combined = combine_block_statistics(runs)
combined = runs{1}.block_statistics;
combined.sigma_min_counts(:) = 0;
combined.cond2_counts(:) = 0;
combined.total_blocks = 0;
combined.global_min_sigma_min = Inf;
combined.global_min_location = empty_block_location();
combined.global_max_block_cond2 = -Inf;
combined.global_max_location = empty_block_location();
for j=1:numel(runs)
    stats = runs{j}.block_statistics;
    combined.sigma_min_counts = combined.sigma_min_counts+stats.sigma_min_counts;
    combined.cond2_counts = combined.cond2_counts+stats.cond2_counts;
    combined.total_blocks = combined.total_blocks+stats.total_blocks;
    if stats.global_min_sigma_min<combined.global_min_sigma_min
        combined.global_min_sigma_min = stats.global_min_sigma_min;
        combined.global_min_location = stats.global_min_location;
    end
    if stats.global_max_block_cond2>combined.global_max_block_cond2
        combined.global_max_block_cond2 = stats.global_max_block_cond2;
        combined.global_max_location = stats.global_max_location;
    end
end
combined.sigma_min_fractions = combined.sigma_min_counts/combined.total_blocks;
combined.cond2_fractions = combined.cond2_counts/combined.total_blocks;
worst = runs{1}.worst_full_system;
for j=2:numel(runs)
    if runs{j}.worst_full_system.cond2>worst.cond2
        worst = runs{j}.worst_full_system;
    end
end
combined.worst_full_system = worst;
end

function comparison = compare_results(actual,expected,repetition)
scalar_fields = {'objective_value','relative_objective_error','G_value', ...
    'feasibility_error','l1_feasibility','best_objective'};
exact_fields = [scalar_fields,{'iterations','exit_reason','output_source', ...
    'v_out','best_consensus','final_ensemble','final_ensemble_consensus', ...
    'restarts','restart_count','concentration_event_count', ...
    'best_incumbent_update_count'}];
exact = false(size(exact_fields));
for j=1:numel(exact_fields)
    exact(j) = isequaln(actual.(exact_fields{j}),expected.(exact_fields{j}));
end
absolute_difference = zeros(1,numel(scalar_fields));
relative_difference = zeros(1,numel(scalar_fields));
for j=1:numel(scalar_fields)
    a = actual.(scalar_fields{j});
    b = expected.(scalar_fields{j});
    absolute_difference(j) = abs(a-b);
    relative_difference(j) = abs(a-b)/max(abs(b),realmin);
end
comparison = struct('repetition',repetition,'fields',{exact_fields}, ...
    'exact_equal',exact,'all_exact',all(exact), ...
    'scalar_fields',{scalar_fields}, ...
    'absolute_differences',absolute_difference, ...
    'relative_differences',relative_difference,'tolerance_used',0, ...
    'pass',all(exact));
end

function output = compact_final_output(result)
fields = {'objective_value','objective_error','relative_objective_error', ...
    'G_value','feasibility_error','l1_feasibility','iterations', ...
    'exit_reason','output_source','v_out','best_objective','best_consensus', ...
    'final_ensemble_consensus','restarts','restart_count', ...
    'concentration_event_count','best_incumbent_update_count', ...
    'runtime','initial_seed','solver_seed','repetition'};
output = struct();
for j=1:numel(fields)
    output.(fields{j}) = result.(fields{j});
end
end

function reference = reference_values(q)
reference = struct();
reference.r = 1;
reference.r2 = 1;
reference.lambda_tangential = 1;
reference.lambda_radial = 1+8*q;
reference.block_cond2 = max(abs([1 1+8*q]))/min(abs([1 1+8*q]));
tangential_r2 = 1-1/(4*q);
radial_r2 = (4*q-1)/(12*q);
reference.tangential_singular_r2 = tangential_r2;
reference.tangential_singular_r = sqrt(tangential_r2);
reference.radial_singular_r2 = radial_r2;
reference.radial_singular_r = sqrt(radial_r2);
reference.tangential_singularity_positive = tangential_r2>0;
reference.radial_singularity_positive = radial_r2>0;
end

function values = values_at_percentiles(data,percentiles)
values = arrayfun(@(p)percentile_linear(data,p),percentiles);
end

function value = percentile_linear(data,p)
data = sort(data(~isnan(data)));
assert(~isempty(data));
position = 1+(numel(data)-1)*(p/100);
lower = floor(position);
upper = ceil(position);
if lower==upper
    value = data(lower);
else
    fraction = position-lower;
    value = data(lower)*(1-fraction)+data(upper)*fraction;
end
end

function location = block_location(repetition,iteration,particle,electron,r2,lambda_tan,lambda_rad,sigma_min,sigma_max,condition)
location = struct('repetition',repetition,'iteration',iteration, ...
    'particle',particle,'free_electron_index',electron,'r2',r2, ...
    'lambda_tangential',lambda_tan,'lambda_radial',lambda_rad, ...
    'sigma_min',sigma_min,'sigma_max',sigma_max,'cond2',condition);
end

function location = empty_block_location()
location = block_location(NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN);
end

function location = empty_full_location()
location = struct('repetition',NaN,'iteration',NaN,'particle',NaN, ...
    'sigma_min',NaN,'sigma_max',NaN,'cond2',-Inf, ...
    'min_singular_block',empty_block_location(), ...
    'max_singular_block',empty_block_location());
end
