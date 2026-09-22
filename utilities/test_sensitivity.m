function tests = test_sensitivity
%TEST_SENSITIVITY Parameter sensitivity, reference values, and restart noise.
tests = functiontests(localfunctions);
end

function setupOnce(testCase)
root = fileparts(fileparts(mfilename('fullpath')));
run(fullfile(root,'setup_repo.m'));
testCase.TestData.root = root;
end

function testSensitivityGridCompletenessAndBaseline(testCase)
config = config_sensitivity();
verifyEqual(testCase,config.case_id,4);
verifyEqual(testCase,config.dimension,20);
verifyEqual(testCase,config.alpha_values,[10,20,30,50,80]);
verifyEqual(testCase,config.epsilon_values,[0.005,0.01,0.02,0.05,0.1]);
verifySize(testCase,config.cells,[5,5]);
pairs = [[config.cells.alpha].',[config.cells.epsilon].'];
verifyEqual(testCase,size(unique(pairs,'rows'),1),25);
verifyTrue(testCase,ismember([50,0.01],pairs,'rows'));
verifyEqual(testCase,config.alpha_over_epsilon, ...
    config.alpha_values(:)./config.epsilon_values,'AbsTol',0);

benchmark = get_revision_benchmark_config(4);
baseline = config.cells(config.alpha_values==50, ...
    config.epsilon_values==0.01).solver_config;
verifyEqual(testCase,baseline,benchmark.solver_config);
end

function testOnlyAlphaAndEpsilonVary(testCase)
config = config_sensitivity();
first = config.cells(1,5).solver_config;
second = config.cells(5,1).solver_config;
verifyNotEqual(testCase,first.alpha,second.alpha);
verifyNotEqual(testCase,first.epsilon,second.epsilon);
verifyEqual(testCase,rmfield(first,{'alpha','epsilon'}), ...
    rmfield(second,{'alpha','epsilon'}));
end

function testSensitivityCompactFieldsAndSeedPairing(testCase)
scratch = make_scratch();
cleanup = onCleanup(@()rmdir(scratch,'s'));
config = make_fast_sensitivity_config();
options = struct('repetitions',1,'pairs',[10,0.1;80,0.005], ...
    'output_file',fullfile(scratch,'shared.mat'));
experiment = run_sensitivity(config,options);
weak = experiment.results{1,5,1};
strong = experiment.results{5,1,1};
verifyEqual(testCase,weak.initial_seed,strong.initial_seed);
verifyEqual(testCase,weak.solver_seed,strong.solver_seed);
verifyEqual(testCase,experiment.storage_mode,'compact');
verifyFalse(testCase,isfield(experiment,'base_initializations'));
verifyFalse(testCase,isfield(experiment,'resolved_solver_configs'));
verifyFalse(testCase,isfield(experiment,'full_intended_config'));
verifyEqual(testCase,experiment.initial_seeds(1),weak.initial_seed);
verifyEqual(testCase,experiment.solver_seeds(1),weak.solver_seed);
required = {'alpha','epsilon','alpha_over_epsilon','seed','initial_seed', ...
    'stochastic_seed','solver_seed','repetition','success','v_out', ...
    'objective_value','objective_gap', ...
    'G_value','feasibility_error','distance_to_vstar','iterations', ...
    'exit_reason','output_source','concentration_event_count', ...
    'restart_count','best_incumbent_update_count','runtime', ...
    'runtime_seconds','all_finite','method','problem_name','benchmark_id', ...
    'case_id','alpha_index','epsilon_index','sigma_indep'};
verifyTrue(testCase,all(isfield(weak,required)));
heavy = {'trajectory','active_mask','initial_ensemble','initial_particles', ...
    'final_ensemble','final_weights','final_ensemble_consensus','config', ...
    'best_consensus'};
verifyFalse(testCase,any(isfield(weak,heavy)));
base_config = validate_solver_config(config.problem, ...
    config.baseline_solver_config);
rng(weak.initial_seed,'twister');
base_config.initial_particles = [];
regenerated = initialize_particles(config.problem,base_config);
verifySize(testCase,regenerated,[config.dimension,base_config.particles]);
verifyEqual(testCase,regenerated,regenerate_initialization( ...
    config.problem,base_config,weak.initial_seed),'AbsTol',0);
summary = summarize_alpha_epsilon_sensitivity(experiment);
verifyEqual(testCase,summary.completed_count(1,5),1);
verifyEqual(testCase,summary.completed_count(5,1),1);
verifyEqual(testCase,summary.completed_count(1,1),0);
clear cleanup
end

function testSensitivityParameterPropagation(testCase)
problem = get_revision_benchmark_config(4).problem;
config = default_solver_config();
config.particles = 3;
config.lambda = 0.7;
config.sigma = 0;
config.gamma = 0.1;
V = reshape(linspace(-0.4,0.7,60),20,3);
Z = zeros(size(V));

low_alpha = config; low_alpha.alpha = 10; low_alpha.epsilon = 0.01;
high_alpha = low_alpha; high_alpha.alpha = 80;
[~,low_diagnostics] = constrained_cbo_step(V,problem,low_alpha,Z);
[~,high_diagnostics] = constrained_cbo_step(V,problem,high_alpha,Z);
expected_low = compute_consensus(V,problem.E,10);
expected_high = compute_consensus(V,problem.E,80);
verifyEqual(testCase,low_diagnostics.v_alpha,expected_low,'AbsTol',0);
verifyEqual(testCase,high_diagnostics.v_alpha,expected_high,'AbsTol',0);
verifyGreaterThan(testCase,norm(low_diagnostics.weights- ...
    high_diagnostics.weights),1e-8);

small_epsilon = config; small_epsilon.alpha = 50; small_epsilon.epsilon = 0.005;
large_epsilon = small_epsilon; large_epsilon.epsilon = 0.1;
actual_small = constrained_cbo_step(V,problem,small_epsilon,Z);
actual_large = constrained_cbo_step(V,problem,large_epsilon,Z);
verifyGreaterThan(testCase,norm(actual_small-actual_large,'fro'),1e-8);
c = small_epsilon.gamma/small_epsilon.epsilon;
va = compute_consensus(V,problem.E,small_epsilon.alpha);
rhs = small_epsilon.lambda*small_epsilon.gamma*(V(:,1)-va) ...
    +c*problem.gradG(V(:,1));
expected_first = V(:,1)-(eye(20)+c*problem.hessG(V(:,1)))\rhs;
verifyEqual(testCase,actual_small(:,1),expected_first,'AbsTol',2e-14);
end

function testSensitivityBaselineTrajectoryIdentity(testCase)
scratch = make_scratch();
cleanup = onCleanup(@()rmdir(scratch,'s'));
config = config_sensitivity();
for index=1:numel(config.cells)
    config.cells(index).solver_config.max_steps = 2;
end
options = struct('repetitions',1,'pairs',[50,0.01], ...
    'output_file',fullfile(scratch,'baseline.mat'), ...
    'store_full_results',true);
experiment = run_sensitivity(config,options);
from_grid = experiment.results{4,2,1};

initial_seed = derive_repetition_seed(config.master_seed,1,0);
solver_seed = derive_repetition_seed(config.master_seed,1,1);
direct_config = config.baseline_solver_config;
direct_config.max_steps = 2;
rng(initial_seed,'twister');
direct_config.initial_particles = [];
base = initialize_particles(config.problem,direct_config);
direct_config.initial_particles = base;
direct_config.seed = solver_seed;
direct = run_constrained_cbo_independent_noise(config.problem,direct_config);
verifyEqual(testCase,from_grid.initial_ensemble,direct.initial_ensemble,'AbsTol',0);
verifyEqual(testCase,from_grid.final_ensemble,direct.final_ensemble,'AbsTol',0);
verifyEqual(testCase,from_grid.v_out,direct.v_out,'AbsTol',0);
verifyEqual(testCase,from_grid.trajectory,direct.trajectory);
verifyEqual(testCase,from_grid.output_source,direct.output_source);
clear cleanup
end

function testCompactFullNumericalIdentityAndCheckpointSize(testCase)
scratch = make_scratch();
cleanup = onCleanup(@()rmdir(scratch,'s'));
config = config_sensitivity();
config.baseline_solver_config.particles = 400;
for index=1:numel(config.cells)
    config.cells(index).solver_config.particles = 400;
    config.cells(index).solver_config.max_steps = 25;
end
full_file = fullfile(scratch,'full.mat');
compact_file = fullfile(scratch,'compact.mat');
common = struct('repetitions',1,'pairs',[50,0.01],'resume',false);
full_options = common;
full_options.output_file = full_file;
full_options.store_full_results = true;
compact_options = common;
compact_options.output_file = compact_file;
compact_options.store_full_results = false;
full_experiment = run_sensitivity(config,full_options);
compact_experiment = run_sensitivity(config,compact_options);
full = full_experiment.results{4,2,1};
compact = compact_experiment.results{4,2,1};
verifyEqual(testCase,full_experiment.storage_mode,'full');
verifyEqual(testCase,compact_experiment.storage_mode,'compact');
verifyTrue(testCase,isfield(full,'trajectory'));
verifyTrue(testCase,isfield(full,'initial_ensemble'));
verifyTrue(testCase,isfield(full,'final_ensemble'));
verifyTrue(testCase,isfield(full_experiment,'base_initializations'));
verifyFalse(testCase,isfield(compact,'trajectory'));
verifyFalse(testCase,isfield(compact,'initial_ensemble'));
verifyFalse(testCase,isfield(compact,'final_ensemble'));
verifyFalse(testCase,isfield(compact_experiment,'base_initializations'));
verify_revision_result_identity(testCase,full,compact);
full_info = dir(full_file);
compact_info = dir(compact_file);
verifyLessThan(testCase,compact_info.bytes,0.5*full_info.bytes);
end

function testStorageModeMismatchRejected(testCase)
scratch = make_scratch();
cleanup = onCleanup(@()rmdir(scratch,'s'));
config = make_fast_sensitivity_config();
output = fullfile(scratch,'mode.mat');
compact = struct('repetitions',1,'pairs',[50,0.01], ...
    'output_file',output,'store_full_results',false);
run_sensitivity(config,compact);
full = compact;
full.store_full_results = true;
verifyError(testCase,@()run_sensitivity(config,full), ...
    'run_sensitivity:CheckpointMismatch');
clear cleanup
end

function testSensitivityCheckpointResume(testCase)
scratch = make_scratch();
cleanup = onCleanup(@()rmdir(scratch,'s'));
config = make_fast_sensitivity_config();
output = fullfile(scratch,'resume.mat');
options = struct('repetitions',2,'pairs',[50,0.01],'output_file',output);
experiment = run_sensitivity(config,options);
preserved = experiment.results{4,2,1}.v_out;
experiment.results{4,2,2} = [];
experiment.completed_mask(4,2,2) = false;
experiment.completed_task_count = 1;
experiment.is_complete = false;
save(output,'experiment','-v7.3');
resumed = run_sensitivity(config,options);
verifyTrue(testCase,resumed.is_complete);
verifyEqual(testCase,resumed.completed_task_count,2);
verifyEqual(testCase,resumed.results{4,2,1}.v_out,preserved,'AbsTol',0);
verifyNotEmpty(testCase,resumed.results{4,2,2});
clear cleanup
end

function testPreciseParaboloidReference(testCase)
problem = ackley_paraboloid(20);
reference = problem.metadata.reference;
verifyLessThanOrEqual(testCase,abs(problem.g(problem.vstar)),1e-14);
verifyEqual(testCase,problem.G(problem.vstar),reference.G_value, ...
    'AbsTol',2e-30);
verifyEqual(testCase,problem.objective_star,problem.E(problem.vstar), ...
    'AbsTol',2e-15);
verifyLessThan(testCase,max(problem.vstar(1:19))- ...
    min(problem.vstar(1:19)),1e-15);

recomputed = recompute_ackley_paraboloid_reference();
verifyGreaterThan(testCase,recomputed.full.exitflag,0);
verifyLessThan(testCase,recomputed.full.firstorderopt,1e-8);
verifyGreaterThan(testCase,recomputed.full.kkt_polish_exitflag,0);
verifyLessThan(testCase,recomputed.full.kkt_residual_inf,1e-12);
verifyLessThan(testCase,recomputed.full.feasibility_error,1e-12);
verifyEqual(testCase,recomputed.full.vstar,problem.vstar,'AbsTol',2e-13);
verifyEqual(testCase,recomputed.full.objective_value, ...
    problem.objective_star,'AbsTol',2e-13);
verifyEqual(testCase,recomputed.symmetric.vstar,problem.vstar, ...
    'AbsTol',2e-13);
verifyEqual(testCase,recomputed.symmetric.objective_value, ...
    problem.objective_star,'AbsTol',2e-13);
verifyLessThan(testCase,abs(recomputed.symmetric.directional_derivative),1e-11);
verifyLessThan(testCase,recomputed.max_coordinate_difference,2e-13);
verifyLessThan(testCase,recomputed.objective_difference,2e-13);
verifyLessThan(testCase,recomputed.full_symmetry_spread,2e-13);

head = problem.vstar(1:19);
directions = {[1;-1;zeros(17,1)]/sqrt(2), ...
    [ones(18,1);-18]/sqrt(342)};
for direction_index=1:numel(directions)
    direction = directions{direction_index};
    for step=[1e-6,1e-5,1e-4]
        nearby_head = head+step*direction;
        nearby = [nearby_head;sum(nearby_head.^2)];
        verifyGreaterThanOrEqual(testCase,problem.E(nearby), ...
            problem.objective_star-2e-14);
    end
end
old = [0.3542*ones(19,1);2.3839];
verifyGreaterThan(testCase,problem.E(old),problem.objective_star);
end

function testReferenceChangeDoesNotChangeTrajectory(testCase)
new_problem = ackley_paraboloid(20);
old_problem = new_problem;
old_problem.vstar = [0.3542*ones(19,1);2.3839];
old_problem.objective_star = old_problem.E(old_problem.vstar);
config = get_revision_benchmark_config(4).solver_config;
config.max_steps = 3;
config.seed = 77701;
rng(77692,'twister');
config.initial_particles = [];
config.initial_particles = initialize_particles(new_problem,config);
old_result = run_constrained_cbo_independent_noise(old_problem,config);
new_result = run_constrained_cbo_independent_noise(new_problem,config);
verifyEqual(testCase,new_result.initial_ensemble,old_result.initial_ensemble, ...
    'AbsTol',0);
verifyEqual(testCase,new_result.final_ensemble,old_result.final_ensemble, ...
    'AbsTol',0);
verifyEqual(testCase,new_result.v_out,old_result.v_out,'AbsTol',0);
verifyEqual(testCase,new_result.objective_value,old_result.objective_value, ...
    'AbsTol',0);
verifyEqual(testCase,new_result.G_value,old_result.G_value,'AbsTol',0);
verifyEqual(testCase,new_result.iterations,old_result.iterations);
verifyEqual(testCase,new_result.exit_reason,old_result.exit_reason);
verifyEqual(testCase,new_result.output_source,old_result.output_source);
verifyNotEqual(testCase,new_result.distance_to_vstar, ...
    old_result.distance_to_vstar);
end

function testAblationFirstRestartIsolationAndBookkeeping(testCase)
problem = zero_constraint_problem();
base = default_solver_config();
base.particles = 2;
base.alpha = 2;
base.epsilon = 0.1;
base.lambda = 0;
base.sigma = 0;
base.gamma = 0.25;
base.max_steps = 1;
base.concentration_tol = Inf;
base.improvement_tol = 0;
base.sigma_indep = 0.4;
base.seed = 9123;
base.initial_particles = [0,0];
zero = base;
zero.sigma_indep = 0;

original_result = run_constrained_cbo_independent_noise(problem,base);
zero_result = run_constrained_cbo_independent_noise(problem,zero);
verifyEqual(testCase,original_result.initial_ensemble, ...
    zero_result.initial_ensemble,'AbsTol',0);
verifyEqual(testCase,original_result.best_consensus, ...
    zero_result.best_consensus,'AbsTol',0);
verifyEqual(testCase,original_result.best_objective, ...
    zero_result.best_objective,'AbsTol',0);

rng(base.seed,'twister');
expected_perturbation = base.sigma_indep*sqrt(base.gamma)*randn(1,2);
verifyEqual(testCase,original_result.final_ensemble-base.initial_particles, ...
    expected_perturbation,'AbsTol',0);
verifyEqual(testCase,zero_result.final_ensemble-zero.initial_particles, ...
    zeros(1,2),'AbsTol',0);
verifyGreaterThan(testCase,norm(expected_perturbation),0);
verifyEqual(testCase,original_result.restart_count,1);
verifyEqual(testCase,zero_result.restart_count,1);
verifyEqual(testCase,original_result.restarts,original_result.restart_count);
verifyEqual(testCase,zero_result.restarts,zero_result.restart_count);
verifyEqual(testCase,original_result.concentration_event_count,2);
verifyEqual(testCase,zero_result.concentration_event_count,2);
verifyEqual(testCase,original_result.best_incumbent_update_count,1);
verifyEqual(testCase,zero_result.best_incumbent_update_count,1);
verifyEqual(testCase,original_result.v_out,zero_result.v_out,'AbsTol',0);
verifyEqual(testCase,original_result.output_source, ...
    'best-concentrated-consensus');
verifyEqual(testCase,zero_result.output_source, ...
    'best-concentrated-consensus');
end

function testAblationFallbackSemanticsBothArms(testCase)
problem = zero_constraint_problem();
base = default_solver_config();
base.particles = 2;
base.max_steps = 1;
base.concentration_tol = -1;
base.initial_particles = [0,2];
base.seed = 9200;
zero = base;
zero.sigma_indep = 0;
original_result = run_constrained_cbo_independent_noise(problem,base);
zero_result = run_constrained_cbo_independent_noise(problem,zero);
verifyEqual(testCase,original_result.output_source,'final-consensus-fallback');
verifyEqual(testCase,zero_result.output_source,'final-consensus-fallback');
verifyEqual(testCase,original_result.restart_count,0);
verifyEqual(testCase,zero_result.restart_count,0);
verifyEmpty(testCase,original_result.best_consensus);
verifyEmpty(testCase,zero_result.best_consensus);
end

function config = make_fast_sensitivity_config()
config = config_sensitivity();
for index=1:numel(config.cells)
    config.cells(index).solver_config.max_steps = 0;
end
end

function regenerated = regenerate_initialization(problem,config,seed)
rng(seed,'twister');
config.initial_particles = [];
regenerated = initialize_particles(problem,config);
end

function verify_revision_result_identity(testCase,full,compact)
exact_fields = {'v_out','objective_value','objective_gap','G_value', ...
    'feasibility_error','distance_to_vstar','success','iterations', ...
    'exit_reason','output_source','restart_count', ...
    'concentration_event_count','best_incumbent_update_count','all_finite'};
for index=1:numel(exact_fields)
    name = exact_fields{index};
    verifyEqual(testCase,compact.(name),full.(name),'AbsTol',0);
end
end

function problem = zero_constraint_problem()
problem = struct('name','zero_constraint_restart_test','dimension',1, ...
    'E',@(V)V.^2,'G',@(V)zeros(1,size(V,2)), ...
    'gradG',@(V)zeros(size(V)),'hessG',@(V)zeros(1,1,size(V,2)), ...
    'g',@(V)zeros(0,size(V,2)),'vstar',0,'objective_star',0, ...
    'metadata',struct('constraint','none'));
end

function scratch = make_scratch()
scratch = tempname;
mkdir(scratch);
end
