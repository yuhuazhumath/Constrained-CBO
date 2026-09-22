function [best_objective,best_consensus,should_stop,objective_gap] = ...
    update_algorithm2_incumbent(current_objective,current_consensus, ...
    best_objective,best_consensus,improvement_tol)
%UPDATE_ALGORITHM2_INCUMBENT Apply the Algorithm 2 E-star decision.
% Measure the stopping gap before replacing an improved incumbent.
objective_gap = abs(current_objective-best_objective);
if current_objective<best_objective
    best_objective = current_objective;
    best_consensus = current_consensus;
end
should_stop = objective_gap<=improvement_tol;
end
