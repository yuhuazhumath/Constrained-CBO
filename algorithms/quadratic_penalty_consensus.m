function [v_alpha,weights,energies] = quadratic_penalty_consensus(V,problem,config)
%QUADRATIC_PENALTY_CONSENSUS Weights from E_epsilon=E+G/epsilon.
penalized = @(X) problem.E(X)+problem.G(X)/config.epsilon;
[v_alpha,weights,energies] = compute_consensus(V,penalized,config.alpha);
end
