function trajectory = trajectory_initialize(problem, config)
%TRAJECTORY_INITIALIZE Allocate trajectory arrays; inactive entries remain NaN.
K = config.max_steps+1;
d = problem.dimension;
trajectory = struct();
if config.store_trajectory
    trajectory.consensus = NaN(d,K);
else
    trajectory.consensus = zeros(d,0);
end
trajectory.objective = NaN(1,K);
trajectory.G = NaN(1,K);
trajectory.feasibility = NaN(1,K);
trajectory.distance = NaN(1,K);
trajectory.concentration = NaN(1,K);
trajectory.active_mask = false(1,K);
trajectory.steps = 0:config.max_steps;
trajectory.store_trajectory = config.store_trajectory;
trajectory.snapshots = cell(1,numel(config.snapshot_steps));
trajectory.snapshot_steps = config.snapshot_steps;
end
