function table2 = make_table2(input_files,output_file)
%MAKE_TABLE2 Long-form Family-B summary from terminal states and current refs.
arguments
    input_files
    output_file char = ''
end
current_configs = match_current_figure6_configs(input_files);
table2 = make_summary_table(input_files,output_file,current_configs);
end

function configs = match_current_figure6_configs(input_files)
files = string(input_files);
files = files(:);
catalog = config_ackley(36000);
indices = zeros(numel(files),1);
for index=1:numel(files)
    loaded = load(files(index),'experiment');
    match = find(string({catalog.id})==string(loaded.experiment.id),1);
    if ~isempty(match), indices(index) = match; end
end
if all(indices==0)
    configs = [];
    return
end
assert(all(indices>0),'make_table2:MixedExperimentFamilies', ...
    'Figure 6 inputs cannot be mixed with other experiment families.');
configs = catalog(indices);
end
