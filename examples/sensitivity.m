% Tables 4 and 7: alpha/epsilon sensitivity for the 20D spherical Ackley case.
run(fullfile(fileparts(fileparts(mfilename('fullpath'))),'setup_repo.m'));
cd(repo_root);
run_sensitivity(config_sensitivity(3,36500));
make_additional_tables("sensitivity",repo_root);
