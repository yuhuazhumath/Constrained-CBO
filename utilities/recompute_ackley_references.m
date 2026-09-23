function references = recompute_ackley_references()
%RECOMPUTE_ACKLEY_REFERENCES Deterministic validation solves for frozen data.
% Experiments use the saved reference values.

segment_problem = ackley_problem(3,0.4*ones(3,1),20,0.1,1);
a = [0.2;0.5;0.7];
u = [0.3;-0.3;-0.2];
phi = @(t) segment_problem.E(a+u*t);
grid_size = 20001;
grid = linspace(0,1,grid_size);
[~,index] = min(phi(grid));
half_width = 10/(grid_size-1);
lower = max(0,grid(index)-half_width);
upper = min(1,grid(index)+half_width);
segment_options = optimset('TolX',1e-14,'MaxFunEvals',10000, ...
    'MaxIter',10000,'Display','off');
[t_fminbnd,~,exitflag,output] = ...
    fminbnd(phi,lower,upper,segment_options);
directional_derivative = @(t) segment_directional_derivative( ...
    a+u*t,u,segment_problem.metadata.ackley);
root_half_width = 20/(grid_size-1);
tstar = fzero(directional_derivative, ...
    [max(0,t_fminbnd-root_half_width),min(1,t_fminbnd+root_half_width)], ...
    optimset('TolX',1e-14,'Display','off'));
vstar = a+u*tstar;
objective_value = phi(tstar);
references.segment = struct('tstar',tstar,'vstar',vstar, ...
    'objective_value',objective_value,'G_value',segment_distance_squared(vstar,a,u), ...
    'feasibility_error',sqrt(segment_distance_squared(vstar,a,u)), ...
    'exitflag',exitflag,'output',output,'grid_size',grid_size, ...
    'fminbnd_tstar',t_fminbnd, ...
    'bracket',[lower,upper]);

d = 20;
m = 18;
radius = 0.5;
embedded_problem = ackley_problem(d,0.4*ones(d,1),20,0.1,1);
parameters = embedded_problem.metadata.ackley;
initial = radius/sqrt(m)*ones(m,1);
embedded_options = optimoptions('fmincon','Display','off', ...
    'Algorithm','interior-point','MaxIterations',2000, ...
    'MaxFunctionEvaluations',100000,'OptimalityTolerance',1e-13, ...
    'ConstraintTolerance',1e-14,'StepTolerance',1e-14, ...
    'FunctionTolerance',1e-14,'SpecifyObjectiveGradient',true, ...
    'SpecifyConstraintGradient',true);
[head,objective_value,exitflag,output,lambda,gradient] = fmincon( ...
    @(y) embedded_objective(y,parameters),initial,[],[],[],[],[],[], ...
    @(y) ball_constraint(y,radius),embedded_options);
vstar = [head;0;0];
constraint_violation = max(sum(head.^2)-radius^2,0);
references.embedded_ball = struct('vstar',vstar, ...
    'objective_value',objective_value,'G_value',constraint_violation^2, ...
    'feasibility_error',constraint_violation,'exitflag',exitflag, ...
    'firstorderopt',output.firstorderopt, ...
    'constraint_violation',output.constrviolation,'output',output, ...
    'lambda',lambda,'gradient',gradient,'initial_point',initial);
end

function [value,gradient] = embedded_objective(y,parameters)
[value,full_gradient] = ackley_objective([y;0;0],parameters);
gradient = full_gradient(1:18);
end

function [c,ceq,gradient_c,gradient_ceq] = ball_constraint(y,radius)
c = sum(y.^2)-radius^2;
ceq = [];
gradient_c = 2*y;
gradient_ceq = [];
end

function value = segment_distance_squared(v,a,u)
t = min(max((u.'*(v-a))/(u.'*u),0),1);
value = sum((v-(a+u*t)).^2);
end

function value = segment_directional_derivative(v,direction,parameters)
[~,gradient] = ackley_objective(v,parameters);
value = direction.'*gradient;
end
