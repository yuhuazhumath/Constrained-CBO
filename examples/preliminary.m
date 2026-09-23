% Figures 1(b) and 2(a-c): preliminary constrained Ackley comparisons.
run(fullfile(fileparts(fileparts(mfilename('fullpath'))),'setup_repo.m'));
cd(repo_root);
run_experiments("figure1_2");
make_figure1b_summary(repo_root, ...
    fullfile('results','tables','paper','figure1b_data.csv'), ...
    fullfile('results','tables','prelim'));
render_manuscript_figures(repo_root,"figure2");
