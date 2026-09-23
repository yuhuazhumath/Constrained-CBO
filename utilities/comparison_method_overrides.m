function overrides = comparison_method_overrides(epsilon,beta,diffusion_type,disable_snapshots)
%COMPARISON_METHOD_OVERRIDES Method-only settings for later comparisons.
arguments
    epsilon (1,1) double {mustBePositive}
    beta (1,1) double {mustBePositive}
    diffusion_type char
    disable_snapshots (1,1) logical = false
end
% Algorithm 2's concentration threshold differs from the penalty stopping tolerance.
penalty = struct('epsilon',epsilon,'concentration_tol',1e-14);
cb2o = struct('beta',beta,'diffusion_type',diffusion_type);
if disable_snapshots
    % Figure 5 needs full-particle snapshots only from the proposed method.
    penalty.snapshot_steps = [];
    cb2o.snapshot_steps = [];
end
overrides = struct( ...
    'method',{'quadratic-penalty-cbo','cb2o'}, ...
    'solver_config',{penalty,cb2o});
end
