% Figure 6 and Table 2: seven constrained Ackley examples.
run(fullfile(fileparts(fileparts(mfilename('fullpath'))),'setup_repo.m'));
cd(repo_root);
run_experiments("figure6");
configs = config_ackley();
make_table2(string({configs.output_file}), ...
    fullfile('results','tables','paper','table2_data.csv'));
render_manuscript_figures(repo_root,"figure6");
