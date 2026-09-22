function catalog = get_paper_experiment_configs(master_seeds)
%GET_PAPER_EXPERIMENT_CONFIGS Build every numerical configuration centrally.
if nargin<1 || isempty(master_seeds)
    master_seeds = struct('figure1_2',12000,'figure4_5',24000, ...
        'figure6',36000,'figure7',47000);
elseif isnumeric(master_seeds)
    validateattributes(master_seeds,{'double'}, ...
        {'scalar','integer','nonnegative'});
    base_seed = master_seeds;
    master_seeds = struct('figure1_2',base_seed, ...
        'figure4_5',base_seed+10000,'figure6',base_seed+20000, ...
        'figure7',base_seed+30000);
end
required = {'figure1_2','figure4_5','figure6','figure7'};
assert(isstruct(master_seeds) && all(isfield(master_seeds,required)), ...
    'Master seeds must be a struct with fields: %s.',strjoin(required,', '));
catalog = struct();
catalog.figure1_2 = config_preliminary(master_seeds.figure1_2);
catalog.figure4_5 = config_simple(master_seeds.figure4_5);
catalog.figure6 = config_ackley(master_seeds.figure6);
catalog.figure7 = config_thomson(master_seeds.figure7);
end
