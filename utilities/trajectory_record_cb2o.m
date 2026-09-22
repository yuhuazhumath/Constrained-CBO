function [trajectory,diagnostics] = trajectory_record_cb2o( ...
    trajectory,step,V,problem,config,stop_concentration)
%TRAJECTORY_RECORD_CB2O Record metrics at a fresh CB2O consensus.
[consensus,~,diagnostics] = compute_cb2o_consensus( ...
    V,problem,config.alpha,config.beta);
idx = step+1;
if trajectory.store_trajectory
    trajectory.consensus(:,idx) = consensus;
end
trajectory.objective(idx) = problem.E(consensus);
trajectory.G(idx) = problem.G(consensus);
trajectory.feasibility(idx) = sqrt(max(trajectory.G(idx),0));
if isempty(problem.vstar)
    trajectory.distance(idx) = NaN;
else
    trajectory.distance(idx) = norm(consensus-problem.vstar)/sqrt(problem.dimension);
end
if nargin<6
    stop_concentration = mean(sum((V-consensus).^2,1))/problem.dimension;
end
trajectory.concentration(idx) = stop_concentration;
trajectory.quantile_count(idx) = diagnostics.quantile_count;
trajectory.quantile_threshold(idx) = diagnostics.quantile_threshold;
trajectory.active_mask(idx) = true;
snapshot_index = find(trajectory.snapshot_steps==step,1);
if ~isempty(snapshot_index)
    trajectory.snapshots{snapshot_index} = V;
end
end
