function V_next = cb2o_step(V,consensus,config,Z)
%CB2O_STEP Euler-Maruyama update from Equation (5.1) of the CB2O reference.
arguments
    V double
    consensus double
    config struct
    Z double
end
assert(isequal(size(Z),size(V)),'cb2o_step:NoiseSize', ...
    'Z must have the same size as V.');
assert(isequal(size(consensus),[size(V,1),1]), ...
    'cb2o_step:ConsensusSize','Consensus must be a column vector.');

diff = V-consensus;
switch char(config.diffusion_type)
    case 'anisotropic'
        diffusion = diff.*Z;
    case 'isotropic'
        diffusion = sqrt(sum(diff.^2,1)).*Z;
    otherwise
        error('cb2o_step:UnknownDiffusion', ...
            'Unknown CB2O diffusion type "%s".',char(config.diffusion_type));
end
V_next = V-config.lambda*config.gamma*diff ...
    +config.sigma*sqrt(config.gamma)*diffusion;
end
