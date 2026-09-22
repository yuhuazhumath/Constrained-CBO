function [steps,values,active] = collect_method_trajectory(experiment,method,field)
%COLLECT_METHOD_TRAJECTORY Assemble NaN-backed repetitions for one method.
method_index = find(strcmp(experiment.methods,method),1);
assert(~isempty(method_index),'collect_method_trajectory:MissingMethod', ...
    'Experiment %s has no %s result.',experiment.id,method);
runs = experiment.results(method_index,:);
first = find(~cellfun(@isempty,runs),1);
assert(~isempty(first),'collect_method_trajectory:NoRuns', ...
    'Experiment %s has no completed %s run.',experiment.id,method);
steps = runs{first}.trajectory.steps(:);
values = NaN(numel(steps),numel(runs));
active = false(numel(steps),numel(runs));
for repetition=1:numel(runs)
    if isempty(runs{repetition}), continue; end
    trajectory = runs{repetition}.trajectory;
    assert(isequal(trajectory.steps(:),steps), ...
        'collect_method_trajectory:StepMismatch', ...
        'Trajectory grids differ within %s.',experiment.id);
    field_values = trajectory.(field);
    values(:,repetition) = field_values(:);
    active(:,repetition) = trajectory.active_mask(:);
end
values(~active) = NaN;
end
