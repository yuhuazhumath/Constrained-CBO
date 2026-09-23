function summary = make_summary_table(input_files,output_file,current_configs)
%MAKE_SUMMARY_TABLE One long-form row for every completed case and method.
% Pass current experiment configurations to recompute reference-dependent
% terminal metrics from v_out instead of accepting checkpoint-derived values.
arguments
    input_files
    output_file char = ''
    current_configs = []
end
files = string(input_files);
files = files(:);
if ~isempty(current_configs)
    assert(isstruct(current_configs) && numel(current_configs)==numel(files), ...
        'make_summary_table:InvalidCurrentConfigs', ...
        'Provide one current configuration for every input MAT file.');
end
experiments = cell(numel(files),1);
row_count = 0;
for file_index=1:numel(files)
    loaded = load(files(file_index),'experiment');
    experiments{file_index} = loaded.experiment;
    row_count = row_count+numel(loaded.experiment.methods);
end
rows = cell(row_count,22);
row_index = 0;
for file_index=1:numel(files)
    experiment = experiments{file_index};
    current_config = resolve_current_config( ...
        current_configs,file_index,experiment);
    for method_index=1:numel(experiment.methods)
        row_index = row_index+1;
        runs = completed_runs(experiment,method_index);
        if isempty(current_config)
            success = numeric_field(runs,'success');
            distance = numeric_field(runs,'distance_to_vstar');
            objective_gap = numeric_field(runs,'objective_gap','objective_error');
        else
            [success,distance,objective_gap] = current_reference_metrics( ...
                runs,current_config);
        end
        feasibility = numeric_field(runs,'feasibility_error');
        iterations = numeric_field(runs,'iterations');
        runtime = numeric_field(runs,'runtime');
        rows(row_index,:) = {string(experiment.id), ...
            string(experiment.problem_name),constraint_name(experiment), ...
            string(experiment.methods{method_index}), ...
            numel(runs),sum(success,'omitnan'),mean(success,'omitnan'), ...
            mean(distance,'omitnan'),median(distance,'omitnan'),std(distance,'omitnan'), ...
            mean(objective_gap,'omitnan'),median(objective_gap,'omitnan'), ...
            mean(feasibility,'omitnan'),median(feasibility,'omitnan'), ...
            mean(iterations,'omitnan'),median(iterations,'omitnan'), ...
            mean(runtime,'omitnan'),median(runtime,'omitnan'), ...
            count_text(runs,'exit_reason'),count_text(runs,'output_source'), ...
            string(files(file_index)),experiment.master_seed};
    end
end
names = {'case_id','problem_name','constraint','method','completed_repetitions', ...
    'success_count','success_rate','mean_distance_to_vstar', ...
    'median_distance_to_vstar','std_distance_to_vstar', ...
    'mean_objective_gap','median_objective_gap', ...
    'mean_feasibility_error','median_feasibility_error', ...
    'mean_iterations','median_iterations','mean_runtime','median_runtime', ...
    'exit_reason_counts','output_source_counts','source_file','master_seed'};
summary = cell2table(rows,'VariableNames',names);
if ~isempty(output_file)
    parent = fileparts(output_file);
    if ~isempty(parent) && ~isfolder(parent), mkdir(parent); end
    writetable(summary,output_file);
end
end

function config = resolve_current_config(configs,file_index,experiment)
if isempty(configs)
    config = [];
    return
end
config = configs(file_index);
assert(isfield(config,'id') && string(config.id)==string(experiment.id), ...
    'make_summary_table:ConfigurationMismatch', ...
    'Current configuration does not match experiment %s.',experiment.id);
assert(isfield(config,'problem') && isfield(config,'solver_config'), ...
    'make_summary_table:IncompleteCurrentConfig', ...
    'Current configuration for %s lacks problem or solver metadata.', ...
    experiment.id);
end

function [success,distance,objective_gap] = current_reference_metrics(runs,config)
n = numel(runs);
success = false(1,n);
distance = NaN(1,n);
objective_gap = NaN(1,n);
for index=1:n
    assert(isfield(runs{index},'v_out') && ~isempty(runs{index}.v_out), ...
        'make_summary_table:MissingTerminalState', ...
        'Cannot recompute current metrics without a saved v_out.');
    metrics = compute_metrics( ...
        config.problem,runs{index}.v_out,config.solver_config);
    success(index) = metrics.success;
    distance(index) = metrics.distance_to_vstar;
    objective_gap(index) = metrics.objective_error;
end
end

function name = constraint_name(experiment)
name = "";
if isfield(experiment,'problem_metadata') ...
        && isfield(experiment.problem_metadata,'constraint')
    name = string(experiment.problem_metadata.constraint);
end
end

function runs = completed_runs(experiment,method_index)
runs = experiment.results(method_index,:);
present = ~cellfun(@isempty,runs);
if isfield(experiment,'completed_mask')
    present = present & experiment.completed_mask(method_index,:);
end
runs = runs(present);
end

function values = numeric_field(runs,name,fallback)
if nargin<3, fallback = ''; end
values = NaN(1,numel(runs));
for index=1:numel(runs)
    if isfield(runs{index},name)
        values(index) = double(runs{index}.(name));
    elseif ~isempty(fallback) && isfield(runs{index},fallback)
        values(index) = double(runs{index}.(fallback));
    end
end
end

function text = count_text(runs,name)
values = strings(1,0);
for index=1:numel(runs)
    if isfield(runs{index},name) && ~isempty(runs{index}.(name))
        values(end+1) = string(runs{index}.(name)); %#ok<AGROW>
    end
end
if isempty(values)
    text = "";
    return
end
unique_values = unique(values,'stable');
parts = strings(size(unique_values));
for index=1:numel(unique_values)
    parts(index) = unique_values(index)+"="+sum(values==unique_values(index));
end
text = strjoin(parts,";");
end
