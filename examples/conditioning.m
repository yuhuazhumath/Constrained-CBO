% Table 6: implicit-system conditioning in k=470 Thomson repetitions 1:3.
run(fullfile(fileparts(fileparts(mfilename('fullpath'))),'setup_repo.m'));
cd(repo_root);
configs = config_thomson();
run_experiment(configs(arrayfun(@(c)c.problem.metadata.k==470,configs)));
diagnostic_file = fullfile('results','diagnostics','conditioning', ...
    'k470_conditioning_reps1_3.mat');
if ~isfile(diagnostic_file)
    run_conditioning_k470_diagnostic();
end
make_additional_tables("conditioning",repo_root);
