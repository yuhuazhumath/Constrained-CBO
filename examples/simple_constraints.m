% Figures 4 and 5 and Table 1: segment, ellipse, and line examples.
run(fullfile(fileparts(fileparts(mfilename('fullpath'))),'setup_repo.m'));
cd(repo_root);
run_experiments("figure4_5");
configs = config_simple();
make_table1(string({configs.output_file}), ...
    fullfile('results','tables','paper','table1_data.csv'));
render_manuscript_figures(repo_root,"figure4");
render_manuscript_figures(repo_root,"figure5");
