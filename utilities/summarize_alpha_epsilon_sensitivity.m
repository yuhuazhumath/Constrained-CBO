function summary = summarize_alpha_epsilon_sensitivity(experiment)
%SUMMARIZE_ALPHA_EPSILON_SENSITIVITY Aggregate sensitivity results by parameter pair.
assert(strcmp(experiment.family,'sensitivity-alpha-epsilon'), ...
    'summarize_alpha_epsilon_sensitivity:WrongFamily', ...
    'Expected a saved sensitivity experiment.');
metric_names = {'objective_gap','feasibility_error','distance_to_vstar', ...
    'iterations'};
shape = [numel(experiment.alpha_values),numel(experiment.epsilon_values)];
summary = struct('alpha_values',experiment.alpha_values, ...
    'epsilon_values',experiment.epsilon_values, ...
    'alpha_over_epsilon',experiment.alpha_over_epsilon, ...
    'axis_semantics',experiment.axis_semantics, ...
    'completed_count',zeros(shape),'success_rate',NaN(shape));
for metric_index=1:numel(metric_names)
    name = metric_names{metric_index};
    summary.(['mean_',name]) = NaN(shape);
    summary.(['median_',name]) = NaN(shape);
end
for alpha_index=1:shape(1)
    for epsilon_index=1:shape(2)
        results = experiment.results(alpha_index,epsilon_index,:);
        results = results(~cellfun(@isempty,results));
        summary.completed_count(alpha_index,epsilon_index) = numel(results);
        if isempty(results)
            continue
        end
        summary.success_rate(alpha_index,epsilon_index) = mean( ...
            cellfun(@(result)double(result.success),results));
        for metric_index=1:numel(metric_names)
            name = metric_names{metric_index};
            values = cellfun(@(result)result.(name),results);
            summary.(['mean_',name])(alpha_index,epsilon_index) = ...
                mean(values,'omitnan');
            summary.(['median_',name])(alpha_index,epsilon_index) = ...
                median(values,'omitnan');
        end
    end
end
end
