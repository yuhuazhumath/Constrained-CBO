function config = config_noise_ablation(family_seed)
%CONFIG_NOISE_ABLATION Table 5 k=56 zero-noise configuration.
% The sigma_indep=0.3 arm is the canonical Figure 7 k=56 configuration.
arguments
    family_seed (1,1) double {mustBeInteger,mustBeNonnegative} = 47000
end
configs = config_thomson(family_seed);
matches = arrayfun(@(item)item.problem.metadata.k==56,configs);
assert(nnz(matches)==1,'Expected exactly one k=56 Thomson configuration.');
config = configs(matches);
config.id = 'figure7_thomson_k56_sigmaindep0';
config.solver_config.sigma_indep = 0;
config.output_file = fullfile('results','raw','revision','indep_noise', ...
    sprintf('ablation_thomson_k56_sigmaindep0_seed%d.mat', ...
    config.master_seed));
end
