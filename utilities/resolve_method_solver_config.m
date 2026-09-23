function solver_config = resolve_method_solver_config(config,method)
%RESOLVE_METHOD_SOLVER_CONFIG Merge a method-only override into the base.
arguments
    config struct
    method char
end
solver_config = config.solver_config;
if ~isfield(config,'method_overrides') || isempty(config.method_overrides)
    return
end
matches = strcmp({config.method_overrides.method},method);
assert(sum(matches)<=1,'resolve_method_solver_config:DuplicateOverride', ...
    'Method %s has more than one override.',method);
if ~any(matches)
    return
end
override = config.method_overrides(matches).solver_config;
names = fieldnames(override);
for index=1:numel(names)
    solver_config.(names{index}) = override.(names{index});
end
end
