% Table 8: median per-iteration times and median paired overhead ratios.
run(fullfile(fileparts(fileparts(mfilename('fullpath'))),'setup_repo.m'));
cd(repo_root);
run_experiments(["figure4_5","figure6"]);
make_additional_tables("runtime",repo_root);
