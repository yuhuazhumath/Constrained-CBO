function [V_next,diagnostics] = projected_cbo_step(V,problem,config,Z)
%PROJECTED_CBO_STEP Euler-Maruyama step for sphere-constrained KV-CBO.
% Input particles must already be feasible. The update uses the tangent
% projector, scalar stochastic amplitude, Ito correction, then normalization.
arguments
    V double
    problem struct
    config struct
    Z double
end
assert(isequal(size(V),size(Z)));
[v_alpha,weights,energies] = compute_consensus(V,problem.E,config.alpha);
d = size(V,1);
n = size(V,2);
V_next = zeros(size(V));
projectors = zeros(d,d,n);
for j=1:n
    v = V(:,j);
    norm2 = v.'*v;
    assert(norm2>0,'projected_cbo_step:ZeroParticle', ...
        'Projected CBO cannot project a zero particle.');
    P = eye(d)-(v*v.')/norm2;
    diff = v-v_alpha;
    dB = sqrt(config.gamma)*Z(:,j);
    ito = -(config.gamma*config.sigma^2/2) ...
        *(diff.'*diff)*(d-1)*v/norm2;
    candidate = v+config.lambda*config.gamma*P*v_alpha ...
        +config.sigma*norm(diff)*P*dB+ito;
    V_next(:,j) = candidate/norm(candidate);
    projectors(:,:,j) = P;
end
diagnostics = struct('v_alpha',v_alpha,'weights',weights, ...
    'energies',energies,'projectors',projectors,'Z',Z);
end
