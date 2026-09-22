% Table 5: paired independent-noise ablation for the k=56 Thomson problem.
run(fullfile(fileparts(fileparts(mfilename('fullpath'))),'setup_repo.m'));
cd(repo_root);
configs = config_thomson();
run_experiment(configs(arrayfun(@(c)c.problem.metadata.k==56,configs)));
run_experiment(config_noise_ablation(47000));
make_additional_tables("ablation",repo_root);
