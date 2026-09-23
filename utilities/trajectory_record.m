function trajectory = trajectory_record(trajectory, step, V, problem, config, objective)
%TRAJECTORY_RECORD Record state V after exactly STEP completed updates.
if nargin<6
    objective = problem.E;
end
[v_alpha,~,~] = compute_consensus(V,objective,config.alpha);
idx = step+1;
if trajectory.store_trajectory
    trajectory.consensus(:,idx) = v_alpha;
end
trajectory.objective(idx) = problem.E(v_alpha);
trajectory.G(idx) = problem.G(v_alpha);
trajectory.feasibility(idx) = sqrt(max(trajectory.G(idx),0));
if isempty(problem.vstar)
    trajectory.distance(idx) = NaN;
else
    trajectory.distance(idx) = norm(v_alpha-problem.vstar)/sqrt(problem.dimension);
end
trajectory.concentration(idx) = mean(sum((V-v_alpha).^2,1))/problem.dimension;
trajectory.active_mask(idx) = true;
snapshot_index = find(trajectory.snapshot_steps==step,1);
if ~isempty(snapshot_index)
    trajectory.snapshots{snapshot_index} = V;
end
end
