function config = validate_solver_config(problem, config)
%VALIDATE_SOLVER_CONFIG Fill optional fields and validate common settings.
defaults = default_solver_config();
fields = fieldnames(defaults);
for i=1:numel(fields)
    if ~isfield(config,fields{i})
        config.(fields{i}) = defaults.(fields{i});
    end
end
assert(config.particles>=1 && config.particles==floor(config.particles));
assert(config.max_steps>=0 && config.max_steps==floor(config.max_steps));
assert(config.alpha>=0 && config.epsilon>0 && config.gamma>0);
assert(config.lambda>=0 && config.sigma>=0);
assert(islogical(config.store_trajectory) && isscalar(config.store_trajectory), ...
    'validate_solver_config:InvalidStoreTrajectory', ...
    'store_trajectory must be a scalar logical value.');
if ~isempty(config.initial_particles)
    assert(isequal(size(config.initial_particles), ...
        [problem.dimension,config.particles]), ...
        'validate_solver_config:InitialParticleSize', ...
        'initial_particles must have size problem.dimension-by-particles.');
end
config.snapshot_steps = unique(config.snapshot_steps(:).');
assert(all(config.snapshot_steps>=0 & config.snapshot_steps<=config.max_steps), ...
    'validate_solver_config:InvalidSnapshotStep', ...
    'Snapshot steps must lie between 0 and max_steps.');
end
