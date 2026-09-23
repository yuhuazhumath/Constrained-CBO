function [consensus,weights,diagnostics] = compute_cb2o_consensus(V,problem,alpha,beta)
%COMPUTE_CB2O_CONSENSUS Quantile-selected CB2O consensus.
% Notation in Equations (5.2)-(5.3) of the CB2O reference:
%   G_CB2O (upper objective) = problem.E
%   L_CB2O (lower objective) = problem.G
% problem.G is the constraint potential (g(v)^2 for a scalar equality).

arguments
    V double
    problem struct
    alpha (1,1) double {mustBeNonnegative}
    beta (1,1) double {mustBePositive,mustBeLessThanOrEqual(beta,1)}
end

N = size(V,2);
assert(N>=1,'compute_cb2o_consensus:EmptyEnsemble', ...
    'CB2O requires at least one particle.');
lower_values = reshape(problem.G(V),1,[]);
assert(numel(lower_values)==N,'compute_cb2o_consensus:LowerObjectiveSize', ...
    'problem.G must return one lower-objective value per particle.');
ordered_lower = sort(lower_values,'ascend');
order_index = ceil(beta*N);
quantile_threshold = ordered_lower(order_index);
selected_mask = lower_values<=quantile_threshold;
selected_indices = find(selected_mask);

% Weight the selected particles by E, shifting by its minimum for stability.
upper_selected = reshape(problem.E(V(:,selected_mask)),1,[]);
shift = min(upper_selected);
selected_weights = exp(-alpha*(upper_selected-shift));
selected_weights = selected_weights/sum(selected_weights);
weights = zeros(1,N);
weights(selected_mask) = selected_weights;
consensus = V*weights.';

diagnostics = struct();
diagnostics.lower_values = lower_values;
diagnostics.upper_selected = upper_selected;
diagnostics.quantile_order_index = order_index;
diagnostics.quantile_threshold = quantile_threshold;
diagnostics.selected_mask = selected_mask;
diagnostics.selected_indices = selected_indices;
diagnostics.quantile_count = numel(selected_indices);
diagnostics.selected_weights = selected_weights;
end
