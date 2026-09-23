% Figure 7 and Table 3: six Thomson problems.
run(fullfile(fileparts(fileparts(mfilename('fullpath'))),'setup_repo.m'));
cd(repo_root);
run_experiments("figure7");
configs = config_thomson();
make_table3(string({configs.output_file}), ...
    fullfile('results','tables','paper','table3_data.csv'));
render_manuscript_figures(repo_root,"figure7");
