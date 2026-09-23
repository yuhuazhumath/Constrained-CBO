function tests = test_solvers
%TEST_SOLVERS Objective, constraint, and solver tests.
tests = functiontests(localfunctions);
end

function setupOnce(testCase)
root = fileparts(fileparts(mfilename('fullpath')));
run(fullfile(root,'setup_repo.m'));
testCase.TestData.root = root;
end

function testAckleyObjective(testCase)
cases = { ...
    struct('V',[0,1;-1,2],'A',20,'a',0.2,'b',3,'vhat',[0.5;-0.2]), ...
    struct('V',[0.1;-0.3;1.2],'A',7,'a',0.4,'b',1.5,'vhat',[0;0;0]), ...
    struct('V',reshape(1:15,5,3)/7,'A',20,'a',0.1,'b',1,'vhat',0.4*ones(5,1))};
for i=1:numel(cases)
    c = cases{i};
    p = struct('A',c.A,'a',c.a,'b',c.b,'vhat',c.vhat);
    actual = ackley_objective(c.V,p);
    d = size(c.V,1);
    delta = c.V-c.vhat;
    expected = -c.A*exp(-c.a*sqrt((c.b^2/d)*sum(delta.^2,1))) ...
        -exp(sum(cos(2*pi*c.b*delta),1)/d)+exp(1)+c.A;
    verifyEqual(testCase,actual,expected,'AbsTol',1e-14);
    [~,gradient] = ackley_objective(c.V(:,1),p);
    gradient_fd = centered_gradient(@(x)ackley_objective(x,p),c.V(:,1),1e-6);
    verifyEqual(testCase,gradient,gradient_fd,'AbsTol',2e-7);
end
end

function testPreciseAckleyReferences(testCase)
segment = ackley_segment();
embedded = ackley_ball();
segment_reference = segment.metadata.reference;
embedded_reference = embedded.metadata.reference;

verifyLessThanOrEqual(testCase,segment.G(segment.vstar),1e-28);
verifyLessThanOrEqual(testCase,embedded.G(embedded.vstar),1e-28);
verifyLessThanOrEqual(testCase,norm(embedded.vstar(1:18))-0.5,1e-15);
verifyEqual(testCase,embedded.vstar(19:20),zeros(2,1),'AbsTol',0);
verifyEqual(testCase,segment.objective_star,segment.E(segment.vstar),'AbsTol',2e-15);
verifyEqual(testCase,embedded.objective_star,embedded.E(embedded.vstar),'AbsTol',2e-14);
verifyEqual(testCase,sqrt(segment.G(segment.vstar)), ...
    segment_reference.feasibility_error,'AbsTol',1e-14);
verifyEqual(testCase,sqrt(embedded.G(embedded.vstar)), ...
    embedded_reference.feasibility_error,'AbsTol',1e-14);

recomputed = recompute_ackley_references();
verifyEqual(testCase,recomputed.segment.tstar,segment_reference.tstar,'AbsTol',2e-13);
verifyEqual(testCase,recomputed.segment.vstar,segment.vstar,'AbsTol',1e-13);
verifyEqual(testCase,recomputed.segment.objective_value, ...
    segment.objective_star,'AbsTol',2e-14);
verifyGreaterThan(testCase,recomputed.embedded_ball.exitflag,0);
verifyLessThan(testCase,recomputed.embedded_ball.firstorderopt,1e-10);
verifyLessThanOrEqual(testCase,recomputed.embedded_ball.constraint_violation,1e-13);
verifyEqual(testCase,recomputed.embedded_ball.vstar,embedded.vstar,'AbsTol',1e-11);
verifyEqual(testCase,recomputed.embedded_ball.objective_value, ...
    embedded.objective_star,'AbsTol',2e-12);

head = embedded.vstar(1:18);
verifyLessThan(testCase,max(head)-min(head),1e-15);
direction = [1;-1;zeros(16,1)]/sqrt(2);
for step = [1e-6,1e-5,1e-4]
    nearby_head = head+step*direction;
    nearby_head = 0.5*nearby_head/norm(nearby_head);
    nearby = [nearby_head;0;0];
    verifyGreaterThanOrEqual(testCase,embedded.E(nearby), ...
        embedded.objective_star-2e-14);
end
end

function testSmoothConstraintDerivatives(testCase)
problems = {ackley_sphere(3),quadratic_ellipse(), ...
    quadratic_line(),ackley_paraboloid(3), ...
    ackley_planes()};
points = {[0.3;-0.4;0.8],[0.2;0.4],[0.4;1.2], ...
    [0.2;-0.3;0.7],[0.1;0.2;0.5]};
for i=1:numel(problems)
    problem = problems{i};
    x = points{i};
    grad_fd = centered_gradient(problem.G,x,1e-6);
    hess_fd = centered_jacobian(problem.gradG,x,2e-5);
    verifyEqual(testCase,problem.gradG(x),grad_fd,'AbsTol',2e-6);
    verifyEqual(testCase,problem.hessG(x),reshape(hess_fd, ...
        problem.dimension,problem.dimension,1),'AbsTol',2e-4);
end
end

function testSegmentSquaredDistance(testCase)
problem = quadratic_segment();
a = problem.metadata.endpoint_a;
b = problem.metadata.endpoint_b;
u = b-a;
e = u/norm(u);
normal = [u(2);-u(1);0];
normal = normal/norm(normal);
points = {(a+b)/2+0.3*normal,a-u,b+u};
expected_project = {(a+b)/2,a,b};
expected_hess = {2*(eye(3)-e*e.'),2*eye(3),2*eye(3)};
for i=1:3
    x = points{i};
    residual = x-expected_project{i};
    verifyEqual(testCase,problem.G(x),sum(residual.^2),'AbsTol',1e-13);
    verifyEqual(testCase,problem.gradG(x),2*residual,'AbsTol',1e-13);
    verifyEqual(testCase,problem.hessG(x),reshape(expected_hess{i},3,3,1), ...
        'AbsTol',1e-13);
end
verifyEqual(testCase,problem.hessG(a),reshape(2*eye(3),3,3,1),'AbsTol',0);
verifyEqual(testCase,problem.hessG(b),reshape(2*eye(3),3,3,1),'AbsTol',0);
end

function testEmbeddedBallSquaredDistance(testCase)
problem = ackley_ball();
m = 18;
d = 20;
inside = zeros(d,1); inside(1)=0.2;
boundary = zeros(d,1); boundary(1)=0.5; boundary(19:20)=[0.2;-0.1];
outside = zeros(d,1); outside(1)=0.8; outside(19:20)=[0.2;-0.1];
verifyEqual(testCase,problem.G(inside),0,'AbsTol',0);
verifyEqual(testCase,problem.gradG(inside),zeros(d,1),'AbsTol',0);
Hinside = zeros(d); Hinside(m+1:end,m+1:end)=2*eye(2);
verifyEqual(testCase,problem.hessG(inside),reshape(Hinside,d,d,1),'AbsTol',0);
verifyEqual(testCase,problem.G(boundary),0.05,'AbsTol',1e-14);
verifyEqual(testCase,problem.gradG(boundary),[zeros(m,1);0.4;-0.2], ...
    'AbsTol',1e-14);
verifyEqual(testCase,problem.hessG(boundary),reshape(Hinside,d,d,1),'AbsTol',0);
projected = outside; projected(1)=0.5; projected(19:20)=0;
verifyEqual(testCase,problem.G(outside),sum((outside-projected).^2),'AbsTol',1e-14);
verifyEqual(testCase,problem.gradG(outside),2*(outside-projected),'AbsTol',1e-14);
r=0.8; z=outside(1:m); expected=Hinside;
expected(1:m,1:m)=2*((1-0.5/r)*eye(m)+0.5/r^3*(z*z.'));
verifyEqual(testCase,problem.hessG(outside),reshape(expected,d,d,1),'AbsTol',1e-13);
end

function testEquation33OneStep(testCase)
problem = quadratic_ellipse();
config = small_config();
config.alpha=2; config.epsilon=0.3; config.lambda=0.7;
config.sigma=0.4; config.gamma=0.05;
V=[0.2,-0.7;0.4,1.1]; Z=[0.3,-0.2;1.2,0.5];
[actual,diagout]=constrained_cbo_step(V,problem,config,Z);
[va]=compute_consensus(V,problem.E,config.alpha);
grad=problem.gradG(V); H=problem.hessG(V); expected=zeros(size(V));
for j=1:2
    lhs=eye(2)+(config.gamma/config.epsilon)*H(:,:,j);
    diff=V(:,j)-va;
    rhs=config.lambda*config.gamma*diff ...
        +(config.gamma/config.epsilon)*grad(:,j) ...
        +config.sigma*sqrt(config.gamma)*diff.*Z(:,j);
    expected(:,j)=V(:,j)-lhs\rhs;
end
verifyEqual(testCase,diagout.v_alpha,va,'AbsTol',1e-15);
verifyEqual(testCase,actual,expected,'AbsTol',5e-15);
end

function testAffineConstraintSolve(testCase)
problem=quadratic_line(); config=small_config();
config.lambda=0; config.sigma=0; config.gamma=0.2; config.epsilon=0.4;
V=[0.4;0.6]; Z=zeros(2,1);
actual=constrained_cbo_step(V,problem,config,Z);
expected=V-(eye(2)+(config.gamma/config.epsilon)*problem.hessG(V))\ ...
    ((config.gamma/config.epsilon)*problem.gradG(V));
verifyEqual(testCase,actual,expected,'AbsTol',1e-14);
end

function testConsensusUsesOnlyE(testCase)
problem=struct('dimension',1,'E',@(V)V,'G',@(V)(V-2).^2, ...
    'gradG',@(V)zeros(size(V)),'hessG',@(V)zeros(1,1,size(V,2)));
config=small_config(); config.alpha=3; config.gamma=0.01;
V=[0,2]; Z=zeros(size(V));
[~,diagout]=constrained_cbo_step(V,problem,config,Z);
[expected]=compute_consensus(V,problem.E,config.alpha);
penalized=@(X)problem.E(X)+problem.G(X)/config.epsilon;
[wrong]=compute_consensus(V,penalized,config.alpha);
verifyEqual(testCase,diagout.v_alpha,expected,'AbsTol',1e-15);
verifyGreaterThan(testCase,abs(diagout.v_alpha-wrong),0.1);
end

function testQuadraticPenaltyWeights(testCase)
problem=quadratic_line(); config=small_config();
config.alpha=1.7; config.epsilon=0.4; V=[0,2,3;0,1,2];
[va,w,e]=quadratic_penalty_consensus(V,problem,config);
energy=problem.E(V)+problem.G(V)/config.epsilon;
shift=exp(-config.alpha*(energy-min(energy))); expectedw=shift/sum(shift);
verifyEqual(testCase,e,energy,'AbsTol',1e-14);
verifyEqual(testCase,w,expectedw,'AbsTol',1e-14);
verifyEqual(testCase,va,V*expectedw.','AbsTol',1e-14);
end

function testProjectedCBO(testCase)
problem=ackley_problem(2,[0;0],20,0.2,3);
config=small_config(); config.alpha=1; config.gamma=0.03;
config.lambda=0.8; config.sigma=0.5;
V=[1,0;0,1]; Z=[0.3,-0.2;0.4,0.7];
[actual,diagout]=projected_cbo_step(V,problem,config,Z);
[va]=compute_consensus(V,problem.E,config.alpha); expected=zeros(size(V));
for j=1:2
    v=V(:,j); P=eye(2)-v*v.'/(v.'*v); diff=v-va;
    candidate=v+config.lambda*config.gamma*P*va ...
        +config.sigma*norm(diff)*P*(sqrt(config.gamma)*Z(:,j)) ...
        -(config.gamma*config.sigma^2/2)*(diff.'*diff)*(2-1)*v/(v.'*v);
    expected(:,j)=candidate/norm(candidate);
    verifyEqual(testCase,P,P.','AbsTol',1e-15);
    verifyEqual(testCase,P*v,zeros(2,1),'AbsTol',1e-15);
end
verifyEqual(testCase,diagout.v_alpha,va,'AbsTol',1e-15);
verifyEqual(testCase,actual,expected,'AbsTol',2e-15);
verifyEqual(testCase,sqrt(sum(actual.^2,1)),ones(1,2),'AbsTol',2e-15);
end

function testThomsonReducedProblem(testCase)
problem=thomson_problem(3);
verifyEqual(testCase,problem.dimension,6);
verifyEqual(testCase,problem.num_constraints,2);
verifyEqual(testCase,problem.nominal_electron_count,3);
verifyEqual(testCase,problem.num_free_electrons,2);
verifyEqual(testCase,problem.fixed_electron,[1;0;0],'AbsTol',0);
X=[1,0,-1;0,1,0;0,0,0];
Vfree=X(:,2:end); V=Vfree(:);
direct=0;
for i=1:2
    for j=i+1:3
        direct=direct+1/norm(X(:,i)-X(:,j));
    end
end
direct=direct/3;
verifyEqual(testCase,problem.E(V),direct,'AbsTol',1e-15);
verifyEqual(testCase,problem.objective_star,1.732050808/3,'AbsTol',1e-15);
verifyEqual(testCase,problem.reconstruct_full(V),reshape(X,3,3,1),'AbsTol',0);
verifyEqual(testCase,problem.g(V),zeros(2,1),'AbsTol',0);
verifyEqual(testCase,problem.G(V),0,'AbsTol',0);
verifyEqual(testCase,problem.gradG(V),zeros(6,1),'AbsTol',0);
Vc=[sqrt(1.2);0;0;sqrt(0.8);0;0];
config=small_config(); metrics=compute_metrics(problem,Vc,config);
verifyEqual(testCase,problem.g(Vc),[0.2;-0.2],'AbsTol',2e-15);
expected_gradient=[4*0.2*sqrt(1.2);0;0;4*(-0.2)*sqrt(0.8);0;0];
verifyEqual(testCase,problem.gradG(Vc),expected_gradient,'AbsTol',3e-15);
expected_hessian=blkdiag(8*diag([1.2,0,0])+4*0.2*eye(3), ...
    8*diag([0.8,0,0])-4*0.2*eye(3));
verifyEqual(testCase,full(problem.hessG(Vc)),expected_hessian,'AbsTol',4e-15);
verifyEqual(testCase,metrics.l1_feasibility,0.4,'AbsTol',3e-15);
verifyEqual(testCase,metrics.feasibility_error,sqrt(0.08),'AbsTol',3e-15);
verifyGreaterThan(testCase,metrics.feasibility_error,0);
end

function testThomsonAngularInitializationAndRestart(testCase)
problem=thomson_problem(3); config=small_config();
config.particles=5; config.initialization=struct('type','thomson_angular');
rng(config.seed,'twister'); actual=initialize_particles(problem,config);
rng(config.seed,'twister');
theta=pi*rand(1,2,config.particles);
phi=2*pi*rand(1,2,config.particles);
expected=reshape([sin(theta).*cos(phi);sin(theta).*sin(phi);cos(theta)], ...
    problem.dimension,config.particles);
verifyEqual(testCase,actual,expected,'AbsTol',0);
norms=sqrt(sum(reshape(actual,3,2,config.particles).^2,1));
verifyEqual(testCase,norms,ones(size(norms)),'AbsTol',3e-15);
full=problem.reconstruct_full(actual);
verifyEqual(testCase,full(:,1,:),repmat([1;0;0],1,1,config.particles),'AbsTol',0);

restart_config=config; restart_config.max_steps=1;
restart_config.concentration_tol=Inf; restart_config.lambda=0;
restart_config.sigma=0; restart_config.sigma_indep=0.3;
result=run_constrained_cbo_independent_noise(problem,restart_config);
verifyEqual(testCase,result.restarts,1);
verifyGreaterThan(testCase,norm(result.final_ensemble-result.initial_ensemble,'fro'),0);
initial_full=problem.reconstruct_full(result.initial_ensemble);
final_full=problem.reconstruct_full(result.final_ensemble);
verifyEqual(testCase,initial_full(:,1,:),repmat([1;0;0],1,1,config.particles),'AbsTol',0);
verifyEqual(testCase,final_full(:,1,:),repmat([1;0;0],1,1,config.particles),'AbsTol',0);
end

function testAlgorithm2BestSoFarStopping(testCase)
best_objective=Inf; best_consensus=[]; tolerance=0.01;
objectives=[1,3,3.005,1.004];
stops=false(size(objectives)); gaps=NaN(size(objectives));
for i=1:numel(objectives)
    [best_objective,best_consensus,stops(i),gaps(i)] = ...
        update_algorithm2_incumbent(objectives(i),i,best_objective, ...
        best_consensus,tolerance);
end
verifyEqual(testCase,stops,[false,false,false,true]);
verifyGreaterThan(testCase,gaps(3),tolerance);
verifyLessThan(testCase,abs(objectives(3)-objectives(2)),tolerance);
verifyEqual(testCase,best_objective,1,'AbsTol',0);
verifyEqual(testCase,best_consensus,1,'AbsTol',0);

problem=struct('name','algorithm2_output_regression','dimension',1, ...
    'E',@(V)1+V.^2,'G',@(V)zeros(1,size(V,2)), ...
    'gradG',@(V)zeros(size(V)),'hessG',@(V)zeros(1,1,size(V,2)), ...
    'g',@(V)zeros(0,size(V,2)),'vstar',0,'objective_star',1);
config=small_config(); config.particles=2; config.max_steps=1;
config.concentration_tol=Inf; config.lambda=0; config.sigma=0;
config.sigma_indep=0.5; config.initial_particles=zeros(1,2);
initial_consensus=compute_consensus(config.initial_particles,problem.E,config.alpha);
result=run_constrained_cbo_independent_noise(problem,config);
final_consensus=compute_consensus(result.final_ensemble,problem.E,config.alpha);
verifyEqual(testCase,result.best_objective,problem.E(initial_consensus),'AbsTol',1e-14);
verifyEqual(testCase,result.best_consensus,initial_consensus,'AbsTol',1e-14);
verifyEqual(testCase,result.v_out,initial_consensus,'AbsTol',1e-14);
verifyGreaterThan(testCase,norm(final_consensus-result.v_out),1e-8);
verifyEqual(testCase,result.final_ensemble_consensus,final_consensus,'AbsTol',0);
verifyEqual(testCase,result.objective_value,problem.E(initial_consensus),'AbsTol',0);
verifyEqual(testCase,result.objective_error,0,'AbsTol',0);
verifyEqual(testCase,result.relative_objective_error,0,'AbsTol',0);
verifyEqual(testCase,result.G_value,problem.G(initial_consensus),'AbsTol',0);
verifyEqual(testCase,result.feasibility_error,0,'AbsTol',0);
verifyEqual(testCase,result.distance_to_vstar,0,'AbsTol',0);
verifyEqual(testCase,result.l1_feasibility,0,'AbsTol',0);
verifyTrue(testCase,result.success);
verifyEqual(testCase,result.output_source,'best-concentrated-consensus');
end

function testAlgorithm2IncumbentSurvivesNonconcentratedTerminal(testCase)
problem=struct('name','algorithm2_terminal_incumbent_regression', ...
    'dimension',1,'E',@(V)V.^2,'G',@(V)zeros(1,size(V,2)), ...
    'gradG',@(V)zeros(size(V)),'hessG',@(V)zeros(1,1,size(V,2)), ...
    'g',@(V)zeros(0,size(V,2)),'vstar',1,'objective_star',1);
config=small_config(); config.particles=2; config.max_steps=1;
config.concentration_tol=1e-14; config.lambda=0; config.sigma=0;
config.gamma=1; config.sigma_indep=1; config.initial_particles=ones(1,2);
candidate_A=compute_consensus(config.initial_particles,problem.E,config.alpha);
result=run_constrained_cbo_independent_noise(problem,config);
terminal_consensus=compute_consensus( ...
    result.final_ensemble,problem.E,config.alpha);

verifyEqual(testCase,result.restarts,1);
verifyGreaterThan(testCase,result.trajectory.concentration(end), ...
    config.concentration_tol);
verifyLessThan(testCase,problem.E(terminal_consensus),problem.E(candidate_A));
verifyEqual(testCase,result.best_consensus,candidate_A,'AbsTol',0);
verifyEqual(testCase,result.best_objective,problem.E(candidate_A),'AbsTol',0);
verifyEqual(testCase,result.v_out,candidate_A,'AbsTol',0);
verifyEqual(testCase,result.final_ensemble_consensus,terminal_consensus,'AbsTol',0);
verifyEqual(testCase,result.objective_value,problem.E(candidate_A),'AbsTol',0);
verifyEqual(testCase,result.objective_error,0,'AbsTol',0);
verifyEqual(testCase,result.relative_objective_error,0,'AbsTol',0);
verifyEqual(testCase,result.G_value,problem.G(candidate_A),'AbsTol',0);
verifyEqual(testCase,result.feasibility_error,0,'AbsTol',0);
verifyEqual(testCase,result.distance_to_vstar,0,'AbsTol',0);
verifyEqual(testCase,result.infinity_distance_to_vstar,0,'AbsTol',0);
verifyEqual(testCase,result.l1_feasibility,0,'AbsTol',0);
verifyTrue(testCase,result.success);
verifyEqual(testCase,result.exit_reason,'max_steps');
verifyEqual(testCase,result.output_source,'best-concentrated-consensus');
end

function testAlgorithm2FinalConsensusFallbackWithoutConcentration(testCase)
problem=struct('name','algorithm2_terminal_fallback_regression', ...
    'dimension',1,'E',@(V)1+V.^2,'G',@(V)zeros(1,size(V,2)), ...
    'gradG',@(V)zeros(size(V)),'hessG',@(V)zeros(1,1,size(V,2)), ...
    'g',@(V)zeros(0,size(V,2)),'vstar',0,'objective_star',1);
config=small_config(); config.particles=2; config.max_steps=1;
config.concentration_tol=-1; config.lambda=0; config.sigma=0;
config.initial_particles=[0,2];
result=run_constrained_cbo_independent_noise(problem,config);
expected=compute_consensus(result.final_ensemble,problem.E,config.alpha);

verifyTrue(testCase,all(result.trajectory.concentration( ...
    result.trajectory.active_mask)>config.concentration_tol));
verifyEqual(testCase,result.restarts,0);
verifyEmpty(testCase,result.best_consensus);
verifyEqual(testCase,result.best_objective,Inf);
verifyEqual(testCase,result.v_out,expected,'AbsTol',0);
verifyEqual(testCase,result.final_ensemble_consensus,expected,'AbsTol',0);
verifyTrue(testCase,all(isfinite(result.v_out)));
verifyEqual(testCase,result.objective_value,problem.E(expected),'AbsTol',0);
verifyEqual(testCase,result.objective_error, ...
    abs(problem.E(expected)-problem.objective_star),'AbsTol',0);
verifyEqual(testCase,result.relative_objective_error, ...
    result.objective_error/abs(problem.objective_star),'AbsTol',0);
verifyEqual(testCase,result.G_value,problem.G(expected),'AbsTol',0);
verifyEqual(testCase,result.feasibility_error,0,'AbsTol',0);
verifyEqual(testCase,result.distance_to_vstar,abs(expected),'AbsTol',0);
verifyEqual(testCase,result.infinity_distance_to_vstar,abs(expected),'AbsTol',0);
verifyEqual(testCase,result.l1_feasibility,0,'AbsTol',0);
verifyTrue(testCase,result.success);
verifyEqual(testCase,result.exit_reason,'max_steps');
verifyEqual(testCase,result.output_source,'final-consensus-fallback');
end

function testThomsonHessianVectorFiniteDifference(testCase)
problem=thomson_problem(3);
x=[0.2;-0.4;0.7;-0.6;0.3;0.5];
direction=[0.3;-0.1;0.4;-0.2;0.6;-0.5];
h=1e-5;
finite_difference=(problem.gradG(x+h*direction) ...
    -problem.gradG(x-h*direction))/(2*h);
analytical=problem.hessG(x)*direction;
verifyEqual(testCase,analytical,finite_difference,'AbsTol',2e-8);
end

function testThomsonStructuredSolveMatchesDense(testCase)
problem=thomson_problem(3);
V=[0.2,-0.4;0.5,0.1;-0.3,0.7;0.8,-0.2;-0.1,0.6;0.4,-0.5];
rhs=reshape(1:12,6,2)/13;
coefficient=0.37;
actual=problem.solve_implicit_matrix(V,rhs,coefficient);
expected=zeros(size(V));
for p=1:size(V,2)
    H=full(problem.hessG(V(:,p)));
    expected(:,p)=(eye(problem.dimension)+coefficient*H)\rhs(:,p);
end
verifyEqual(testCase,actual,expected,'AbsTol',5e-14);

config=small_config(); Z=zeros(size(V));
[~,diagnostics]=constrained_cbo_step(V,problem,config,Z);
verifyEqual(testCase,diagnostics.implicit_solver,'problem-specific');
verifyEmpty(testCase,diagnostics.hessG);
end

function testThomsonObjectiveMatchesDirectPairs(testCase)
problem=thomson_problem(3);
V=[0.2,-0.4;0.5,0.1;-0.3,0.7;0.8,-0.2;-0.1,0.6;0.4,-0.5];
expected=zeros(1,size(V,2));
for p=1:size(V,2)
    X=problem.reconstruct_full(V(:,p));
    for i=1:2
        for j=i+1:3
            expected(p)=expected(p)+1/norm(X(:,i)-X(:,j));
        end
    end
end
expected=expected/3;
verifyEqual(testCase,problem.E(V),expected,'AbsTol',2e-14);
verifyEqual(testCase,problem.E(V),legacy_obj_eg3(V),'AbsTol',2e-14);

k_values=[2,3,8,15,56,470];
reference_totals=[0.5,1.732050808,19.675287861,80.670244114, ...
    1337.094945276,104822.886324279];
for i=1:numel(k_values)
    reference_problem=thomson_problem(k_values(i));
    verifyEqual(testCase,reference_problem.objective_star, ...
        reference_totals(i)/k_values(i),'AbsTol',0);
end
end

function testFinalOutputUsesFinalEnsemble(testCase)
problem=quadratic_line(); config=small_config();
config.particles=2; config.max_steps=1; config.concentration_tol=-1;
config.sigma=0; config.initial_particles=[0,1;0,2];
result=run_constrained_cbo(problem,config);
[expected]=compute_consensus(result.final_ensemble,problem.E,config.alpha);
[initial]=compute_consensus(result.initial_ensemble,problem.E,config.alpha);
verifyEqual(testCase,result.v_out,expected,'AbsTol',1e-14);
verifyEqual(testCase,result.final_ensemble_consensus,expected,'AbsTol',0);
verifyGreaterThan(testCase,norm(result.v_out-initial),1e-8);
verifyEqual(testCase,result.objective_value,problem.E(expected),'AbsTol',1e-14);
verifyEqual(testCase,result.G_value,problem.G(expected),'AbsTol',1e-14);
verifyEqual(testCase,result.output_source,'fresh-final-ensemble-consensus');
end

function testReproducibility(testCase)
problem=quadratic_line(); config=small_config();
config.particles=4; config.max_steps=3; config.initial_particles=[];
first=run_constrained_cbo(problem,config);
second=run_constrained_cbo(problem,config);
verifyEqual(testCase,first.initial_ensemble,second.initial_ensemble,'AbsTol',0);
verifyEqual(testCase,first.final_ensemble,second.final_ensemble,'AbsTol',0);
different=config; different.seed=config.seed+1;
third=run_constrained_cbo(problem,different);
verifyGreaterThan(testCase,norm(first.initial_ensemble-third.initial_ensemble,'fro'),0);
end

function testSnapshotIndices(testCase)
problem=quadratic_line(); config=small_config();
config.particles=3; config.max_steps=100; config.concentration_tol=-1;
config.sigma=0; config.snapshot_steps=[0,5,50,100];
config.initial_particles=[0,1,2;1,0,-1];
result=run_constrained_cbo(problem,config);
V=config.initial_particles;
targets=config.snapshot_steps; expected=cell(size(targets)); expected{1}=V;
for step=1:100
    V=constrained_cbo_step(V,problem,config,zeros(size(V)));
    idx=find(targets==step,1);
    if ~isempty(idx), expected{idx}=V; end
end
for i=1:numel(targets)
    verifyEqual(testCase,result.trajectory.snapshots{i},expected{i},'AbsTol',1e-13);
end
end

function testTrajectoryStorageCanBeDisabled(testCase)
problem=quadratic_line(); config=small_config();
config.store_trajectory=false; config.snapshot_steps=[0,2];
result=run_constrained_cbo(problem,config);
verifySize(testCase,result.trajectory.consensus,[problem.dimension,0]);
verifySize(testCase,result.trajectory.objective,[1,config.max_steps+1]);
verifySize(testCase,result.trajectory.concentration,[1,config.max_steps+1]);
verifyTrue(testCase,all(result.trajectory.active_mask));
verifyFalse(testCase,any(cellfun(@isempty,result.trajectory.snapshots)));
end

function testExperimentCheckpointSurvivesLateFailure(testCase)
scratch=tempname; mkdir(scratch);
cleanup=onCleanup(@()rmdir(scratch,'s'));
problem=quadratic_line(); config=small_config();
config.max_steps=0; config.store_trajectory=false;
output_file=fullfile(scratch,'checkpoint.mat');
experiment_config=struct('id','checkpoint_test','problem',problem, ...
    'methods',{{'proposed','deliberately-invalid'}}, ...
    'solver_config',config,'repetitions',1,'master_seed',90100, ...
    'output_file',output_file);
verifyError(testCase,@()run_experiment(experiment_config), ...
    'run_experiment:UnknownMethod');
verifyTrue(testCase,isfile(output_file));
loaded=load(output_file,'experiment');
verifyEqual(testCase,loaded.experiment.completed_mask,[true;false]);
verifyEqual(testCase,loaded.experiment.completed_repetitions,0);
verifyNotEmpty(testCase,loaded.experiment.results{1,1});
verifyEmpty(testCase,loaded.experiment.results{2,1});
verifyFalse(testCase,loaded.experiment.is_complete);
end

function testMissingDataUsesMask(testCase)
values=[0,10;2,20;4,30];
active=[true,false;true,true;false,true];
actual=average_active_trajectory(values,active);
verifyEqual(testCase,actual,[0;11;30],'AbsTol',0);
end

function config=small_config()
config=default_solver_config();
config.particles=2; config.alpha=2; config.epsilon=0.2;
config.lambda=1; config.sigma=0.3; config.gamma=0.05;
config.max_steps=2; config.concentration_tol=-1; config.seed=1234;
config.initialization=struct('type','uniform_box','lower',-1,'upper',1);
end

function grad=centered_gradient(fun,x,h)
d=numel(x); grad=zeros(d,1);
for i=1:d
    e=zeros(d,1); e(i)=1;
    grad(i)=(fun(x+h*e)-fun(x-h*e))/(2*h);
end
end

function J=centered_jacobian(fun,x,h)
d=numel(x); J=zeros(d,d);
for i=1:d
    e=zeros(d,1); e(i)=1;
    J(:,i)=(fun(x+h*e)-fun(x-h*e))/(2*h);
end
end

function values=legacy_obj_eg3(V)
free_electrons=size(V,1)/3;
particles=size(V,2);
W=reshape(V,3,free_electrons,particles);
values=sum(1./sqrt(sum((W-[1;0;0]).^2,1)),2);
for i=2:free_electrons
    values=values+sum(1./sqrt(sum((W(:,1:i-1,:)-W(:,i,:)).^2,1)),2);
end
values=reshape(values,1,particles)/(free_electrons+1);
end
