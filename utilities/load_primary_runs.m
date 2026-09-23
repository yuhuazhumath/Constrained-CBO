function [experiment,runs] = load_primary_runs(input_file)
%LOAD_PRIMARY_RUNS Load the first configured method from a raw result file.
loaded = load(char(input_file),'experiment');
experiment = loaded.experiment;
runs = experiment.results(1,:);
end
