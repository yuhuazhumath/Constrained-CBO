function metadata = revision_solver_config_metadata(config)
%REVISION_SOLVER_CONFIG_METADATA Serializable reconstructable solver config.
% Per-run particles and seeds are supplied by the runner and are therefore
% not embedded in checkpoint signatures or duplicated experiment metadata.
metadata = config;
discard = intersect(fieldnames(metadata),{'initial_particles','seed'});
if ~isempty(discard)
    metadata = rmfield(metadata,discard);
end
end
