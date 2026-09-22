function [v_alpha, weights, energies] = compute_consensus(V, objective, alpha)
%COMPUTE_CONSENSUS Stable Gibbs-weighted consensus using the supplied energy.
arguments
    V double
    objective function_handle
    alpha (1,1) double {mustBeNonnegative}
end
energies = objective(V);
assert(isrow(energies) && numel(energies)==size(V,2), ...
    'compute_consensus:InvalidObjectiveShape', ...
    'The objective must return one value per particle.');
shifted = energies-min(energies);
unnormalized = exp(-alpha*shifted);
normalizer = sum(unnormalized);
assert(isfinite(normalizer) && normalizer>0, ...
    'compute_consensus:InvalidWeights', 'Consensus weights are not finite.');
weights = unnormalized/normalizer;
v_alpha = V*weights.';
end
