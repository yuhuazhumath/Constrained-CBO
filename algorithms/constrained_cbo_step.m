function [V_next, diagnostics] = constrained_cbo_step(V, problem, config, Z)
%CONSTRAINED_CBO_STEP Linearly implicit update from Equation (33).
% Consensus weights use problem.E; the stochastic term has the sign in (33).
arguments
    V double
    problem struct
    config struct
    Z double
end
validate_step_inputs(V,problem,config,Z);
[v_alpha, weights, energies] = compute_consensus(V,problem.E,config.alpha);
d = size(V,1);
n = size(V,2);
diff = V-v_alpha;
gradG = problem.gradG(V);
implicit_coefficient = config.gamma/config.epsilon;
rhs = config.lambda*config.gamma*diff ...
    +implicit_coefficient*gradG ...
    +config.sigma*sqrt(config.gamma)*diff.*Z;
if isfield(problem,'solve_implicit_matrix')
    correction = problem.solve_implicit_matrix(V,rhs,implicit_coefficient);
    hessG = [];
    implicit_solver = 'problem-specific';
else
    hessG = problem.hessG(V);
    correction = zeros(size(V));
    identity = eye(d);
    for j = 1:n
        lhs = identity+implicit_coefficient*hessG(:,:,j);
        correction(:,j) = lhs\rhs(:,j);
    end
    implicit_solver = 'generic-dense';
end
V_next = V-correction;
diagnostics = struct('v_alpha',v_alpha,'weights',weights, ...
    'energies',energies,'gradG',gradG,'hessG',hessG,'Z',Z, ...
    'implicit_solver',implicit_solver);
end

function validate_step_inputs(V,problem,config,Z)
required_problem = {'E','G','gradG','hessG','dimension'};
required_config = {'alpha','epsilon','lambda','sigma','gamma'};
for i=1:numel(required_problem)
    assert(isfield(problem,required_problem{i}), ...
        'constrained_cbo_step:MissingProblemField', ...
        'Problem is missing %s.',required_problem{i});
end
for i=1:numel(required_config)
    assert(isfield(config,required_config{i}), ...
        'constrained_cbo_step:MissingConfigField', ...
        'Configuration is missing %s.',required_config{i});
end
assert(size(V,1)==problem.dimension && isequal(size(V),size(Z)), ...
    'constrained_cbo_step:DimensionMismatch', ...
    'V and Z must be d-by-N arrays matching the problem dimension.');
assert(config.epsilon>0 && config.gamma>0, ...
    'constrained_cbo_step:InvalidStepParameters', ...
    'epsilon and gamma must be positive.');
end
