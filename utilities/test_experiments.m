function tests = test_experiments
%TEST_EXPERIMENTS Settings, shared seeds, summaries, and checkpoint checks.
tests = functiontests(localfunctions);
end

function setupOnce(testCase)
root = fileparts(fileparts(mfilename('fullpath')));
run(fullfile(root,'setup_repo.m'));
testCase.TestData.root = root;
end

function testFamilyConfigurationStructure(testCase)
methods = {'proposed','quadratic-penalty-cbo','cb2o'};
family_a = config_simple(24000);
verifyEqual(testCase,numel(family_a),3);
for index=1:numel(family_a)
    verifyEqual(testCase,family_a(index).methods,methods);
    verifyFalse(testCase,any(strcmp(family_a(index).methods,'projected-cbo')));
    verifyEqual(testCase,family_a(index).proposed_solver,'algorithm1');
    verifyEqual(testCase,family_a(index).solver_config.sigma,1,'AbsTol',0);
    verifyEqual(testCase,family_a(index).repetitions,100);
    verifyEqual(testCase,resolve_method_solver_config( ...
        family_a(index),'proposed'),family_a(index).solver_config);
    penalty = resolve_method_solver_config(family_a(index), ...
        'quadratic-penalty-cbo');
    cb2o = resolve_method_solver_config(family_a(index),'cb2o');
    verifyEqual(testCase,penalty.epsilon,0.01,'AbsTol',0);
    verifyEmpty(testCase,penalty.snapshot_steps);
    verifyEqual(testCase,cb2o.beta,1/20,'AbsTol',0);
    verifyEqual(testCase,cb2o.diffusion_type,'anisotropic');
    verifyEmpty(testCase,cb2o.snapshot_steps);
end

family_b = config_ackley(36000);
verifyEqual(testCase,numel(family_b),7);
for index=1:numel(family_b)
    verifyEqual(testCase,family_b(index).methods,methods);
    verifyFalse(testCase,any(strcmp(family_b(index).methods,'projected-cbo')));
    verifyEqual(testCase,resolve_method_solver_config( ...
        family_b(index),'proposed'),family_b(index).solver_config);
    penalty = resolve_method_solver_config(family_b(index), ...
        'quadratic-penalty-cbo');
    cb2o = resolve_method_solver_config(family_b(index),'cb2o');
    verifyEqual(testCase,penalty.epsilon,0.01,'AbsTol',0);
    verifyEqual(testCase,cb2o.beta,1/20,'AbsTol',0);
    verifyEqual(testCase,cb2o.diffusion_type,'anisotropic');
end
verifyEqual(testCase,{family_b(1:4).proposed_solver}, ...
    repmat({'algorithm1'},1,4));
verifyEqual(testCase,{family_b(5:7).proposed_solver}, ...
    repmat({'algorithm2'},1,3));
verifyEqual(testCase,arrayfun(@(x)x.solver_config.improvement_tol, ...
    family_b(5:7)), ...
    [0.05,0.01,0.001],'AbsTol',0);
verifyEqual(testCase,arrayfun(@(x)x.solver_config.sigma_indep, ...
    family_b(5:7)), ...
    [0.3,0.3,1],'AbsTol',0);
reference = ackley_references("paraboloid_d20");
verifyEqual(testCase,family_b(7).problem.vstar,reference.vstar,'AbsTol',0);
reference = ackley_references("paraboloid_d3");
verifyEqual(testCase,family_b(3).problem.vstar,reference.vstar,'AbsTol',0);
verifyEqual(testCase,family_b(3).problem.objective_star, ...
    reference.objective_value,'AbsTol',0);
end

function testFigureOneTwoProjectedRegression(testCase)
configs = config_preliminary(12000);
verifyEqual(testCase,configs(1).methods, ...
    {'proposed','projected-cbo','quadratic-penalty-cbo','cb2o'});
verifyEqual(testCase,configs(2).methods, ...
    {'proposed','projected-cbo','quadratic-penalty-cbo','cb2o'});
verifyEqual(testCase,configs(3).methods, ...
    {'proposed','quadratic-penalty-cbo','cb2o'});
end

function test20DPenaltyStoppingThresholdIsolation(testCase)
configs = config_ackley(36000);
case4 = configs(7);
base_before = case4.solver_config;
proposed = resolve_method_solver_config(case4,'proposed');
penalty = resolve_method_solver_config(case4,'quadratic-penalty-cbo');
cb2o = resolve_method_solver_config(case4,'cb2o');
verifyEqual(testCase,case4.solver_config.concentration_tol,1e-5,'AbsTol',0);
verifyEqual(testCase,proposed.concentration_tol,1e-5,'AbsTol',0);
verifyEqual(testCase,penalty.concentration_tol,1e-14,'AbsTol',0);
verifyEqual(testCase,penalty.epsilon,0.01,'AbsTol',0);
verifyEqual(testCase,cb2o.beta,1/20,'AbsTol',0);
verifyEqual(testCase,case4.solver_config,base_before);
end

function testCanonicalProblemsAreNotResquared(testCase)
segment = quadratic_segment();
V3 = [0,1,2;0,-1,0;0.5,0,1];
verifyEqual(testCase,segment.G(V3), ...
    sum((V3-segment.project(V3)).^2,1),'AbsTol',0);
ellipse = quadratic_ellipse();
V2 = [-1,0,1;0.5,-0.25,2];
g = (V2(1,:)+1).^2/2+V2(2,:).^2-1;
verifyEqual(testCase,ellipse.G(V2),g.^2,'AbsTol',0);
line = quadratic_line();
g = V2(1,:)+V2(2,:)-3;
verifyEqual(testCase,line.G(V2),g.^2,'AbsTol',0);
verifyEqual(testCase,line.gradG([0;0]),[-6;-6],'AbsTol',0);
end

function testSharedSeedsAndInitializations(testCase)
family_a = config_simple(24100);
verify_paired_run(testCase,family_a(1),0);
family_b = config_ackley(36100);
verify_paired_run(testCase,family_b(1),1);
verify_paired_run(testCase,family_b(7),1);
end

function testMethodOverrideIsolationAndComparatorPropagation(testCase)
config = config_simple(24200);
config = config(3);
proposed_before = resolve_method_solver_config(config,'proposed');
cb2o_before = resolve_method_solver_config(config,'cb2o');
penalty_before = resolve_method_solver_config(config,'quadratic-penalty-cbo');
config.method_overrides(1).solver_config.epsilon = 0.02;
verifyEqual(testCase,resolve_method_solver_config(config,'proposed'), ...
    proposed_before);
verifyEqual(testCase,resolve_method_solver_config(config,'cb2o'),cb2o_before);
penalty_after = resolve_method_solver_config(config,'quadratic-penalty-cbo');
verifyNotEqual(testCase,penalty_after.epsilon,penalty_before.epsilon);
penalty_after_resolved = penalty_after;
V = [-1,0,1,2;0,1,2,3];
penalty_before.alpha = 0.1;
penalty_after.alpha = 0.1;
[~,weights_before] = quadratic_penalty_consensus( ...
    V,config.problem,penalty_before);
[~,weights_after] = quadratic_penalty_consensus( ...
    V,config.problem,penalty_after);
verifyNotEqual(testCase,weights_before,weights_after);

cb_config = config;
cb_config.method_overrides(2).solver_config.beta = 3/4;
verifyEqual(testCase,resolve_method_solver_config(cb_config,'proposed'), ...
    proposed_before);
verifyEqual(testCase,resolve_method_solver_config( ...
    cb_config,'quadratic-penalty-cbo'),penalty_after_resolved);
problem = comparator_mapping_problem();
points = 1:4;
[~,~,small] = compute_cb2o_consensus(points,problem,1,1/4);
[~,~,large] = compute_cb2o_consensus(points,problem,1,3/4);
verifyEqual(testCase,small.quantile_count,1);
verifyEqual(testCase,large.quantile_count,3);
verifyEqual(testCase,small.lower_values,problem.G(points),'AbsTol',0);
verifyEqual(testCase,small.upper_selected,problem.E(points(1)),'AbsTol',0);
verifyNotEqual(testCase,small.lower_values,problem.G(points).^2);
end

function testFigureFiveProductionSnapshotsOnlyProposed(testCase)
config = config_simple(24300);
config = config(3);
config.repetitions = 1;
config.output_file = '';
experiment = run_experiment(config);
proposed = experiment.results{1,1};
verifyEqual(testCase,proposed.trajectory.snapshot_steps,[0,5,50,100]);
verifyFalse(testCase,any(cellfun(@isempty,proposed.trajectory.snapshots)));
for method_index=2:3
    verifyEmpty(testCase,experiment.results{method_index,1}.trajectory.snapshot_steps);
    verifyEmpty(testCase,experiment.results{method_index,1}.trajectory.snapshots);
end
verifyEqual(testCase,proposed.trajectory.snapshots{1}, ...
    proposed.initial_ensemble,'AbsTol',0);
verifyEqual(testCase,proposed.trajectory.consensus(:,6), ...
    compute_consensus(proposed.trajectory.snapshots{2}, ...
    config.problem.E,config.solver_config.alpha),'AbsTol',1e-14);
end

function testUniversalResultSchema(testCase)
config = config_ackley(36200);
config = config(7);
config.repetitions = 1;
config.output_file = '';
config.solver_config.max_steps = 1;
experiment = run_experiment(config);
required = {'case_id','problem_name','method','repetition','initial_seed', ...
    'solver_seed','v_out','objective_value','objective_error','objective_gap', ...
    'G_value','feasibility_error','distance_to_vstar','success','iterations', ...
    'exit_reason','output_source','runtime','trajectory','active_mask', ...
    'initialization_id','all_finite'};
for method_index=1:3
    verifyTrue(testCase,all(isfield(experiment.results{method_index,1},required)));
end
proposed = experiment.results{1,1};
verifyTrue(testCase,all(isfield(proposed,{'concentration_event_count', ...
    'restart_count','best_incumbent_update_count'})));
cb2o = experiment.results{3,1};
verifyTrue(testCase,all(isfield(cb2o,{'beta','diffusion_type', ...
    'min_quantile_count','max_quantile_count','final_quantile_count'})));
end

function testSummaryRowsRemainSeparatedByMethod(testCase)
scratch = make_scratch();
cleanup = onCleanup(@()rmdir(scratch,'s'));
file = fullfile(scratch,'summary.mat');
save_summary_fixture(file);
summary = make_summary_table(file);
verifyEqual(testCase,height(summary),2);
proposed = summary(summary.method=="proposed",:);
penalty = summary(summary.method=="quadratic-penalty-cbo",:);
verifyEqual(testCase,proposed.completed_repetitions,2);
verifyEqual(testCase,proposed.constraint,"fixture-constraint");
verifyEqual(testCase,proposed.success_count,1);
verifyEqual(testCase,proposed.success_rate,0.5,'AbsTol',0);
verifyEqual(testCase,proposed.mean_distance_to_vstar,2,'AbsTol',0);
verifyEqual(testCase,proposed.std_distance_to_vstar,sqrt(2),'AbsTol',1e-14);
verifyEqual(testCase,penalty.success_rate,1,'AbsTol',0);
verifyEqual(testCase,penalty.mean_distance_to_vstar,5,'AbsTol',0);
verifyEqual(testCase,proposed.exit_reason_counts,"max_steps=1;concentration=1");
verifyEqual(testCase,height(make_table1(file)),2);
verifyEqual(testCase,height(make_table2(file)),2);
clear cleanup
end

function testSummaryCanRecomputeCurrentReferenceMetrics(testCase)
scratch = make_scratch();
cleanup = onCleanup(@()rmdir(scratch,'s'));
file = fullfile(scratch,'current_reference_summary.mat');
experiment = struct('id','current_reference_case', ...
    'problem_name','current_reference_problem','methods',{{'proposed'}}, ...
    'repetitions',1,'results',{{struct('v_out',[1.005;2.005], ...
    'success',false,'distance_to_vstar',99,'objective_error',88, ...
    'feasibility_error',0,'iterations',1,'runtime',0.1, ...
    'exit_reason','max_steps','output_source','fixture')}}, ...
    'completed_mask',true,'master_seed',1,'problem_metadata', ...
    struct('constraint','fixture-constraint'));
save(file,'experiment');

problem = struct('name','current_reference_problem','dimension',2, ...
    'E',@(V)sum(V.^2,1),'G',@(V)zeros(1,size(V,2)), ...
    'vstar',[1;2],'objective_star',5);
solver = default_solver_config();
solver.success_infinity_distance_tol = 0.01;
config = struct('id','current_reference_case','problem',problem, ...
    'solver_config',solver);
summary = make_summary_table(file,'',config);

verifyEqual(testCase,summary.success_count,1);
verifyEqual(testCase,summary.mean_distance_to_vstar,0.005,'AbsTol',1e-15);
verifyEqual(testCase,summary.mean_objective_gap,0.03005,'AbsTol',1e-14);
clear cleanup
end

function testTable2UsesCurrentFigure6Reference(testCase)
scratch = make_scratch();
cleanup = onCleanup(@()rmdir(scratch,'s'));
file = fullfile(scratch,'figure6c_current_reference.mat');
config = config_ackley(36000);
config = config(3);
v_out = config.problem.vstar+[0.05;0;0];
run = struct('v_out',v_out,'success',false,'distance_to_vstar',99, ...
    'objective_error',88,'feasibility_error',sqrt(config.problem.G(v_out)), ...
    'iterations',1,'runtime',0.1,'exit_reason','max_steps', ...
    'output_source','fixture');
experiment = struct('id',config.id,'problem_name',config.problem.name, ...
    'methods',{{'proposed'}},'repetitions',1,'results',{{run}}, ...
    'completed_mask',true,'master_seed',config.master_seed, ...
    'problem_metadata',config.problem.metadata);
save(file,'experiment');
summary = make_table2(file);
verifyEqual(testCase,summary.success_count,1);
verifyEqual(testCase,summary.mean_distance_to_vstar, ...
    0.05/sqrt(3),'AbsTol',1e-15);
clear cleanup
end

function testCheckpointResumeMatchesUninterrupted(testCase)
scratch = make_scratch();
cleanup = onCleanup(@()rmdir(scratch,'s'));
config = tiny_comparison_config('');
fresh = run_experiment(config);

partial = fresh;
keep = false(3,2);
keep(1:2,1) = true;
for index=find(~keep).'
    partial.results{index} = [];
end
partial.completed_mask = keep;
partial.completed_task_count = nnz(keep);
partial.completed_repetitions = 0;
partial.is_complete = false;
partial.base_initializations{2} = [];
partial.checkpoint_write_count = 0;
partial.results{1,1}.resume_sentinel = 'must-be-preserved';
output_file = fullfile(scratch,'partial.mat');
partial.output_file = output_file;
save_revision_checkpoint(output_file,partial);

config.output_file = output_file;
resumed = run_experiment(config);
verifyTrue(testCase,resumed.is_complete);
verifyTrue(testCase,resumed.resumed_from_checkpoint);
verifyEqual(testCase,resumed.completed_task_count,6);
verifyEqual(testCase,resumed.completed_repetitions,2);
verifyEqual(testCase,resumed.checkpoint_write_count,2);
verifyEqual(testCase,resumed.results{1,1},partial.results{1,1});
for repetition=1:2
    base = resumed.base_initializations{repetition};
    verifyEqual(testCase,base,fresh.base_initializations{repetition},'AbsTol',0);
    for method_index=1:3
        actual = resumed.results{method_index,repetition};
        expected = fresh.results{method_index,repetition};
        verify_result_numerics(testCase,actual,expected);
        verifyEqual(testCase,actual.initial_seed, ...
            resumed.results{1,repetition}.initial_seed);
        verifyEqual(testCase,actual.solver_seed, ...
            resumed.results{1,repetition}.solver_seed);
        verifyEqual(testCase,actual.initial_ensemble,base,'AbsTol',0);
    end
end
clear cleanup
end

function testCheckpointSignatureMismatchRejected(testCase)
scratch = make_scratch();
cleanup = onCleanup(@()rmdir(scratch,'s'));
output_file = fullfile(scratch,'signature.mat');
config = tiny_comparison_config(output_file);
run_experiment(config);
changed = config;
changed.method_overrides(1).solver_config.epsilon = 0.02;
verifyError(testCase,@()run_experiment(changed), ...
    'run_experiment:CheckpointMismatch');
loaded = load(output_file,'experiment');
verifyEqual(testCase,loaded.experiment.signature, ...
    experiment_checkpoint_signature(config));
clear cleanup
end

function testSingleMethodCheckpointResumeWithMultipleCompletedRuns(testCase)
scratch = make_scratch();
cleanup = onCleanup(@()rmdir(scratch,'s'));
config = tiny_comparison_config('');
config.methods = {'proposed'};
config = rmfield(config,'method_overrides');
config.repetitions = 4;
fresh = run_experiment(config);

partial = fresh;
partial.completed_mask = [true,true,false,false];
partial.results(3:4) = {[]};
partial.base_initializations(3:4) = {[]};
partial.completed_task_count = 2;
partial.completed_repetitions = 2;
partial.is_complete = false;
partial.checkpoint_write_count = 0;
partial.results{1}.resume_sentinel = 'first-preserved';
partial.results{2}.resume_sentinel = 'second-preserved';
output_file = fullfile(scratch,'single-method-partial.mat');
partial.output_file = output_file;
save_revision_checkpoint(output_file,partial);

config.output_file = output_file;
resumed = run_experiment(config);
verifyTrue(testCase,resumed.is_complete);
verifyTrue(testCase,resumed.resumed_from_checkpoint);
verifyEqual(testCase,resumed.completed_mask,true(1,4));
verifyEqual(testCase,resumed.completed_task_count,4);
verifyEqual(testCase,resumed.completed_repetitions,4);
verifyEqual(testCase,resumed.checkpoint_write_count,2);
verifyEqual(testCase,resumed.results{1},partial.results{1});
verifyEqual(testCase,resumed.results{2},partial.results{2});
for repetition=3:4
    verify_result_numerics(testCase,resumed.results{repetition}, ...
        fresh.results{repetition});
end
signature = experiment_checkpoint_signature(config);
verifyEqual(testCase,resumed.initial_seeds,signature.initial_seeds);
verifyEqual(testCase,resumed.solver_seeds,signature.solver_seeds);
for repetition=1:4
    result = resumed.results{repetition};
    verifyEqual(testCase,result.repetition,repetition);
    verifyEqual(testCase,result.initial_seed,resumed.initial_seeds(repetition));
    verifyEqual(testCase,result.solver_seed,resumed.solver_seeds(repetition));
    verifyEqual(testCase,result.stochastic_seed, ...
        resumed.solver_seeds(repetition));
end
clear cleanup
end

function testCheckpointWritesOncePerRepetition(testCase)
scratch = make_scratch();
cleanup = onCleanup(@()rmdir(scratch,'s'));
output_file = fullfile(scratch,'frequency.mat');
config = tiny_comparison_config(output_file);
config.repetitions = 3;
experiment = run_experiment(config);
% One initial checkpoint plus one after each complete repetition.
verifyEqual(testCase,experiment.checkpoint_write_count,4);
first_results = experiment.results;
reused = run_experiment(config);
verifyEqual(testCase,reused.checkpoint_write_count,4);
verifyEqual(testCase,reused.results,first_results);
verifyTrue(testCase,reused.resumed_from_checkpoint);
clear cleanup
end

function testConditioningCacheAcceptsMatchingProduction(testCase)
scratch = make_scratch();
cleanup = onCleanup(@()rmdir(scratch,'s'));
save_conditioning_fixture(scratch);
tables = make_additional_tables("conditioning",scratch);
verifyEqual(testCase,tables.table6.value,[4;8;8;200000;100000;8]);
verifyTrue(testCase,isfile(fullfile(scratch,'results','tables','paper','table6_data.csv')));
clear cleanup
end

function testConditioningCacheRejectsChangedSeeds(testCase)
scratch = make_scratch();
cleanup = onCleanup(@()rmdir(scratch,'s'));
[diagnostic_file,~] = save_conditioning_fixture(scratch);
saved = load(diagnostic_file,'diagnostic');
diagnostic = saved.diagnostic;
diagnostic.initialization_seeds(1) = diagnostic.initialization_seeds(1)+1;
save(diagnostic_file,'diagnostic');
verifyError(testCase,@()make_additional_tables("conditioning",scratch), ...
    'make_additional_tables:ConditioningCacheMismatch');
verifyFalse(testCase,isfile(fullfile(scratch,'results','tables','paper','table6_data.csv')));
clear cleanup
end

function testConditioningCacheRejectsChangedSolverSettings(testCase)
scratch = make_scratch();
cleanup = onCleanup(@()rmdir(scratch,'s'));
[diagnostic_file,~] = save_conditioning_fixture(scratch);
saved = load(diagnostic_file,'diagnostic');
diagnostic = saved.diagnostic;
diagnostic.solver_config.gamma = 2*diagnostic.solver_config.gamma;
save(diagnostic_file,'diagnostic');
verifyError(testCase,@()make_additional_tables("conditioning",scratch), ...
    'make_additional_tables:ConditioningCacheMismatch');
diagnostic = saved.diagnostic;
diagnostic.q = 2*diagnostic.q;
save(diagnostic_file,'diagnostic');
verifyError(testCase,@()make_additional_tables("conditioning",scratch), ...
    'make_additional_tables:ConditioningCacheMismatch');
clear cleanup
end

function testConditioningCacheRejectsChangedProductionSignature(testCase)
scratch = make_scratch();
cleanup = onCleanup(@()rmdir(scratch,'s'));
[~,production_file] = save_conditioning_fixture(scratch);
saved = load(production_file,'experiment');
experiment = saved.experiment;
experiment.signature.base_solver_config.epsilon = 0.02;
save(production_file,'experiment');
verifyError(testCase,@()make_additional_tables("conditioning",scratch), ...
    'make_additional_tables:ConditioningCacheMismatch');
clear cleanup
end

function testConditioningCacheRejectsChangedProductionState(testCase)
scratch = make_scratch();
cleanup = onCleanup(@()rmdir(scratch,'s'));
[~,production_file] = save_conditioning_fixture(scratch);
saved = load(production_file,'experiment');
experiment = saved.experiment;
experiment.results{1,2}.v_out(1) = experiment.results{1,2}.v_out(1)+1;
save(production_file,'experiment');
verifyError(testCase,@()make_additional_tables("conditioning",scratch), ...
    'make_additional_tables:ConditioningCacheMismatch');
clear cleanup
end

function [diagnostic_file,production_file] = save_conditioning_fixture(root)
configs = config_thomson();
config = configs(arrayfun(@(c)c.problem.metadata.k==470,configs));
signature = experiment_checkpoint_signature(config);
experiment = struct('id',config.id,'signature',signature, ...
    'methods',{config.methods},'initial_seeds',signature.initial_seeds, ...
    'solver_seeds',signature.solver_seeds,'repetitions',100, ...
    'completed_repetitions',100,'is_complete',true, ...
    'completed_mask',true(1,100),'results',{cell(1,100)});
for repetition=1:100
    state = repetition*ones(config.problem.dimension,1);
    experiment.results{1,repetition} = struct( ...
        'objective_value',100,'objective_error',1,'relative_objective_error',0.01, ...
        'G_value',1e-8,'feasibility_error',1e-4,'l1_feasibility',1e-3, ...
        'iterations',2000,'exit_reason','max_steps','output_source','best-incumbent', ...
        'v_out',state,'best_objective',100,'best_consensus',state, ...
        'final_ensemble_consensus',state,'restarts',1,'restart_count',1, ...
        'concentration_event_count',2,'best_incumbent_update_count',1, ...
        'runtime',1,'initial_seed',signature.initial_seeds(repetition), ...
        'solver_seed',signature.solver_seeds(repetition),'repetition',repetition);
end
production_file = fullfile(root,config.output_file);
mkdir(fileparts(production_file));
save(production_file,'experiment');
diagnostic = struct('case_id',config.id,'problem_metadata',config.problem.metadata, ...
    'master_seed',config.master_seed,'canonical_repetitions',100, ...
    'repetition_ids',1:3,'initialization_seeds',signature.initial_seeds(1:3), ...
    'solver_seeds',signature.solver_seeds(1:3), ...
    'solver_config',revision_solver_config_metadata(config.solver_config), ...
    'q',config.solver_config.gamma/config.solver_config.epsilon, ...
    'production_file',fullfile('legacy','thomson.mat'), ...
    'reproduction',struct('repetition',{1,2,3},'pass',{true,true,true}), ...
    'run_diagnostics',{cell(1,3)},'final_outputs',{experiment.results(1,1:3)});
minimum_values = [1e-6 1e-5 1];
for repetition=1:3
    diagnostic.run_diagnostics{repetition} = struct( ...
        'repetition',repetition,'iterations',2000, ...
        'full_cond2',2^repetition*ones(2000,50), ...
        'full_sigma_min',minimum_values(repetition)*ones(2000,50));
    diagnostic.final_outputs{repetition}.runtime = 100;
end
diagnostic_file = fullfile(root,'results','diagnostics','conditioning', ...
    'k470_conditioning_reps1_3.mat');
mkdir(fileparts(diagnostic_file));
save(diagnostic_file,'diagnostic');
end

function verify_paired_run(testCase,config,max_steps)
config.repetitions = 1;
config.output_file = '';
config.solver_config.max_steps = max_steps;
config.solver_config.snapshot_steps = [];
experiment = run_experiment(config);
base = experiment.base_initializations{1};
for method_index=1:3
    result = experiment.results{method_index,1};
    verifyEqual(testCase,result.initial_seed, ...
        experiment.results{1,1}.initial_seed);
    verifyEqual(testCase,result.solver_seed, ...
        experiment.results{1,1}.solver_seed);
    verifyEqual(testCase,result.initialization_id, ...
        experiment.results{1,1}.initialization_id);
    verifyEqual(testCase,result.initial_ensemble,base,'AbsTol',0);
end
end

function config = tiny_comparison_config(output_file)
solver = default_solver_config();
solver.particles = 6;
solver.alpha = 2;
solver.epsilon = 0.01;
solver.lambda = 1;
solver.sigma = 0.2;
solver.gamma = 0.05;
solver.max_steps = 3;
solver.concentration_tol = -1;
solver.initialization = struct('type','uniform_box','lower',-1,'upper',1);
config = struct('id','tiny_resume_comparison', ...
    'problem',quadratic_line(), ...
    'methods',{{'proposed','quadratic-penalty-cbo','cb2o'}}, ...
    'solver_config',solver, ...
    'method_overrides',comparison_method_overrides( ...
    solver.epsilon,solver.beta,solver.diffusion_type,false), ...
    'proposed_solver','algorithm1','repetitions',2, ...
    'master_seed',74400,'output_file',output_file);
end

function verify_result_numerics(testCase,actual,expected)
fields = {'v_out','objective_value','objective_error','objective_gap', ...
    'G_value','feasibility_error','distance_to_vstar','success', ...
    'iterations','exit_reason','output_source','initial_seed','solver_seed', ...
    'initial_ensemble','final_ensemble','trajectory','active_mask'};
for index=1:numel(fields)
    name = fields{index};
    verifyEqual(testCase,actual.(name),expected.(name),'AbsTol',0);
end
end

function problem = comparator_mapping_problem()
problem = struct('name','mapping','dimension',1, ...
    'E',@(V)mapping_lookup(V,[4,3,2,1]), ...
    'G',@(V)mapping_lookup(V,[0.1,0.2,0.3,0.4]), ...
    'vstar',1,'objective_star',4);
end

function values = mapping_lookup(V,lookup)
values = lookup(V);
end

function save_summary_fixture(file)
methods = {'proposed','quadratic-penalty-cbo'};
experiment = struct('id','summary_case','problem_name','summary_problem', ...
    'methods',{methods},'repetitions',2,'results',{cell(2,2)}, ...
    'completed_mask',true(2,2),'master_seed',99, ...
    'problem_metadata',struct('constraint','fixture-constraint'));
experiment.results(1,:) = {summary_run(false,1,2,3,4,'max_steps'), ...
    summary_run(true,3,4,5,6,'concentration')};
experiment.results(2,:) = {summary_run(true,4,6,8,10,'max_steps'), ...
    summary_run(true,6,8,10,12,'max_steps')};
save(file,'experiment');
end

function run = summary_run(success,distance,gap,feasibility,iterations,reason)
run = struct('success',success,'distance_to_vstar',distance, ...
    'objective_gap',gap,'feasibility_error',feasibility, ...
    'iterations',iterations,'runtime',iterations/10, ...
    'exit_reason',reason,'output_source','fixture');
end

function scratch = make_scratch()
scratch = tempname;
mkdir(scratch);
end
