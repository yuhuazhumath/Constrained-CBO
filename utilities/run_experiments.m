function manifest = run_experiments(groups,master_seeds)
%RUN_EXPERIMENTS Run the selected paper families, 100 repetitions each.
arguments
    groups string = ["figure1_2","figure4_5","figure6","figure7"]
    master_seeds = []
end
catalog = get_paper_experiment_configs(master_seeds);
allowed = string(fieldnames(catalog));
assert(all(ismember(groups,allowed)),'run_experiments:UnknownGroup', ...
    'Groups must be selected from: %s.',strjoin(allowed,', '));
manifest = struct('group',{},'id',{},'output_file',{});
for group = groups(:).'
    configs = catalog.(group);
    for i = 1:numel(configs)
        run_experiment(configs(i));
        manifest(end+1) = struct('group',group,'id',string(configs(i).id), ...
            'output_file',string(configs(i).output_file)); %#ok<AGROW>
    end
end
end
