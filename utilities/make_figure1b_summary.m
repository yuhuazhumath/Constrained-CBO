function summary = make_figure1b_summary(repo_root,output_file,preliminary_output_dir)
%MAKE_FIGURE1B_SUMMARY Recompute the preliminary summary from terminal states.
arguments
    repo_root char = fileparts(fileparts(mfilename('fullpath')))
    output_file char = ''
    preliminary_output_dir char = ''
end

configs = config_preliminary(12000);
input_files = strings(numel(configs),1);
for index=1:numel(configs)
    input_files(index) = fullfile(repo_root,configs(index).output_file);
end
summary = make_summary_table(input_files,output_file,configs);

if isempty(preliminary_output_dir)
    return
end
if ~isfolder(preliminary_output_dir), mkdir(preliminary_output_dir); end
output_names = ["figure2a_summary.csv","figure2b_summary.csv", ...
    "figure2c_summary.csv"];
for index=1:numel(configs)
    rows = summary.case_id==string(configs(index).id);
    auxiliary = table(summary.method(rows),summary.success_count(rows), ...
        summary.median_distance_to_vstar(rows), ...
        summary.median_objective_gap(rows), ...
        summary.median_feasibility_error(rows), ...
        summary.median_iterations(rows), ...
        'VariableNames',{'method','success_count','median_distance', ...
        'median_objective_error','median_feasibility','median_iterations'});
    writetable(auxiliary,fullfile(preliminary_output_dir,output_names(index)));
end
end
