function tests = test_cb2o
%TEST_CB2O Focused tests for the additive CB2O comparator.
tests = functiontests(localfunctions);
end

function setupOnce(testCase)
root = fileparts(fileparts(mfilename('fullpath')));
run(fullfile(root,'setup_repo.m'));
testCase.TestData.root = root;
end

function testQuantileSelectionIncludesThresholdTies(testCase)
V = 1:5;
problem = bare_problem(1,@(X)zeros(1,size(X,2)), ...
    @(X)lookup_values(X,V,[0.1,0.2,0.2,0.2,0.9]));
[~,weights,diagnostics] = compute_cb2o_consensus(V,problem,1,0.5);
verifyEqual(testCase,diagnostics.quantile_order_index,3);
verifyEqual(testCase,diagnostics.quantile_threshold,0.2,'AbsTol',0);
verifyEqual(testCase,diagnostics.selected_mask,[true,true,true,true,false]);
verifyEqual(testCase,diagnostics.quantile_count,4);
verifyGreaterThan(testCase,diagnostics.quantile_count, ...
    diagnostics.quantile_order_index);
verifyEqual(testCase,find(weights>0),1:4);
end

function testConsensusSelectionAndUpperWeights(testCase)
V = [0,1,2,4];
lower = [9,1,4,1];
upper = [9,1,0,4];
problem = bare_problem(1,@(X)lookup_values(X,V,upper), ...
    @(X)lookup_values(X,V,lower));
alpha = 2;
[consensus,weights,diagnostics] = compute_cb2o_consensus( ...
    V,problem,alpha,0.5);
expected_selected = [false,true,false,true];
expected_selected_weights = exp(-alpha*([1,4]-1));
expected_selected_weights = expected_selected_weights/sum(expected_selected_weights);
expected_weights = [0,expected_selected_weights(1),0,expected_selected_weights(2)];
verifyEqual(testCase,diagnostics.selected_mask,expected_selected);
verifyEqual(testCase,diagnostics.lower_values,lower,'AbsTol',0);
verifyEqual(testCase,diagnostics.upper_selected,[1,4],'AbsTol',0);
verifyEqual(testCase,sum(weights),1,'AbsTol',1e-15);
verifyEqual(testCase,weights,expected_weights,'AbsTol',1e-15);
verifyEqual(testCase,consensus,V*expected_weights.','AbsTol',1e-15);
verifyEqual(testCase,weights(3),0,'AbsTol',0); % Best upper value was not selected.
end

function testStableShiftedWeightsWithLargeOffset(testCase)
V = [-1,0,2];
problem = bare_problem(1,@(X)1e12+(X-1).^2,@(X)zeros(1,size(X,2)));
alpha = 3;
[consensus,weights] = compute_cb2o_consensus(V,problem,alpha,1);
unshifted_energies = (V-1).^2;
expected = exp(-alpha*(unshifted_energies-min(unshifted_energies)));
expected = expected/sum(expected);
verifyTrue(testCase,all(isfinite(weights)));
verifyTrue(testCase,isfinite(consensus));
verifyEqual(testCase,weights,expected,'AbsTol',1e-15);
verifyEqual(testCase,consensus,V*expected.','AbsTol',1e-15);
end

function testOneStepAnisotropicUpdate(testCase)
V = [1,-2,0.5;3,0,-1];
consensus = [0.25;-0.5];
Z = [0.2,-1,0.7;-0.3,0.4,1.1];
config = default_solver_config();
config.lambda = 0.8;
config.sigma = 1.3;
config.gamma = 0.04;
config.diffusion_type = 'anisotropic';
diff = V-consensus;
expected = V-config.lambda*config.gamma*diff ...
    +config.sigma*sqrt(config.gamma)*diff.*Z;
actual = cb2o_step(V,consensus,config,Z);
verifyEqual(testCase,actual,expected,'AbsTol',0);
end

function testDerivativeFreeProblemRuns(testCase)
problem = bare_problem(2,@(V)sum((V-[0.5;-0.2]).^2,1), ...
    @(V)(sum(V.^2,1)-1).^2);
problem.vstar = [1;0];
problem.objective_star = problem.E(problem.vstar);
config = cb2o_test_config();
config.particles = 5;
config.max_steps = 3;
config.initial_particles = [];
result = run_cb2o(problem,config);
verifyTrue(testCase,result.all_finite);
verifyEqual(testCase,result.iterations,3);
verifyFalse(testCase,isfield(problem,'gradG'));
verifyFalse(testCase,isfield(problem,'hessG'));
end

function testFinalConsensusIsFresh(testCase)
problem = bare_problem(1,@(V)(V+0.5).^2,@(V)(V-1).^2);
problem.vstar = 1;
problem.objective_star = problem.E(1);
config = cb2o_test_config();
config.particles = 3;
config.max_steps = 1;
config.lambda = 0.7;
config.sigma = 0;
config.beta = 2/3;
config.alpha = 0.1;
config.initial_particles = [-2,0,3];
[stale] = compute_cb2o_consensus( ...
    config.initial_particles,problem,config.alpha,config.beta);
result = run_cb2o(problem,config);
[expected,weights,diagnostics] = compute_cb2o_consensus( ...
    result.final_ensemble,problem,config.alpha,config.beta);
verifyEqual(testCase,result.v_out,expected,'AbsTol',0);
verifyEqual(testCase,result.final_weights,weights,'AbsTol',0);
verifyEqual(testCase,result.quantile_threshold, ...
    diagnostics.quantile_threshold,'AbsTol',0);
verifyEqual(testCase,result.quantile_count,diagnostics.quantile_count);
verifyGreaterThan(testCase,abs(result.v_out-stale),1e-8);
verifyEqual(testCase,result.output_source,'fresh-final-cb2o-consensus');
end

function testDeterminismAndSeedSensitivity(testCase)
problem = bare_problem(2,@(V)sum((V-[0.2;-0.4]).^2,1), ...
    @(V)(sum(V.^2,1)-1).^2);
problem.vstar = [1;0];
problem.objective_star = problem.E(problem.vstar);
config = cb2o_test_config();
config.particles = 6;
config.max_steps = 4;
config.initial_particles = [];
first = run_cb2o(problem,config);
second = run_cb2o(problem,config);
verifyEqual(testCase,first.initial_ensemble,second.initial_ensemble,'AbsTol',0);
verifyEqual(testCase,first.final_ensemble,second.final_ensemble,'AbsTol',0);
verifyEqual(testCase,first.trajectory.consensus, ...
    second.trajectory.consensus,'AbsTol',0);
different = config;
different.seed = config.seed+1;
third = run_cb2o(problem,different);
verifyGreaterThan(testCase,norm(first.trajectory.consensus ...
    -third.trajectory.consensus,'fro'),0);
end

function testFigure2bObjectiveMapping(testCase)
configs = config_preliminary(12000);
verifyTrue(testCase,all(cellfun(@(methods)any(strcmp(methods,'cb2o')), ...
    {configs.methods})));
problem = configs(2).problem;
canonical = ackley_problem(2,[0.5;1/3],20,0.2,3);
V = [[0;0],[0.5;1/3],problem.vstar];
verifyEqual(testCase,problem.E(V),canonical.E(V),'AbsTol',0);
verifyEqual(testCase,problem.G(V),(sum(V.^2,1)-1).^2,'AbsTol',0);
[~,~,diagnostics] = compute_cb2o_consensus(V,problem,30,1/20);
verifyEqual(testCase,diagnostics.lower_values, ...
    (sum(V.^2,1)-1).^2,'AbsTol',0);
verifyEqual(testCase,diagnostics.upper_selected,problem.E(V(:,3)),'AbsTol',0);
end

function testStoppingStatisticUsesPreUpdateConsensus(testCase)
problem = bare_problem(1,@(V)V.^2,@(V)zeros(1,size(V,2)));
problem.vstar = 0;
problem.objective_star = 0;
config = cb2o_test_config();
config.particles = 2;
config.max_steps = 1;
config.sigma = 0;
config.lambda = 0.4;
config.initial_particles = [-1,2];
initial_consensus = compute_cb2o_consensus( ...
    config.initial_particles,problem,config.alpha,config.beta);
result = run_cb2o(problem,config);
expected = sum((result.final_ensemble-initial_consensus).^2,'all')/2;
verifyEqual(testCase,result.stop_concentration,expected,'AbsTol',0);
verifyEqual(testCase,result.trajectory.concentration(2),expected,'AbsTol',0);
end

function testExperimentUsesSharedBaseInitialization(testCase)
problem = quadratic_line();
solver = cb2o_test_config();
solver.particles = 4;
solver.max_steps = 0;
solver.initial_particles = [];
config = struct('id','cb2o_shared_initialization','problem',problem, ...
    'methods',{{'proposed','cb2o'}},'solver_config',solver, ...
    'repetitions',1,'master_seed',73100);
experiment = run_experiment(config);
base = experiment.base_initializations{1};
verifyEqual(testCase,experiment.results{1,1}.initial_ensemble,base,'AbsTol',0);
verifyEqual(testCase,experiment.results{2,1}.initial_ensemble,base,'AbsTol',0);
verifyEqual(testCase,experiment.results{1,1}.stochastic_seed, ...
    experiment.results{2,1}.stochastic_seed);
end

function problem = bare_problem(d,E,G)
problem = struct('name','cb2o_test_problem','dimension',d,'E',E,'G',G, ...
    'vstar',zeros(d,1),'objective_star',0);
end

function values = lookup_values(X,reference,reference_values)
[found,locations] = ismember(X,reference);
assert(all(found,'all'));
values = reshape(reference_values(locations),1,[]);
end

function config = cb2o_test_config()
config = default_solver_config();
config.particles = 4;
config.alpha = 2;
config.beta = 0.5;
config.lambda = 1;
config.sigma = 0.3;
config.gamma = 0.05;
config.max_steps = 2;
config.epsilon_stop = 0;
config.diffusion_type = 'anisotropic';
config.seed = 2468;
config.initialization = struct('type','uniform_box','lower',-1,'upper',1);
end
