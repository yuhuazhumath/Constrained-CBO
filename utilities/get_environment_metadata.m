function metadata = get_environment_metadata()
%GET_ENVIRONMENT_METADATA Record MATLAB and toolbox versions.
toolboxes = ver;
metadata = struct();
metadata.matlab_release = version('-release');
metadata.matlab_version = version;
metadata.computer = computer;
metadata.toolboxes = rmfield(toolboxes,intersect(fieldnames(toolboxes),{'Date'}));
end
