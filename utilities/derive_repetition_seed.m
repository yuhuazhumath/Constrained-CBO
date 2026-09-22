function seed = derive_repetition_seed(master_seed,repetition,stream_offset)
%DERIVE_REPETITION_SEED Deterministic, recorded seed derivation.
arguments
    master_seed (1,1) double {mustBeInteger,mustBeNonnegative}
    repetition (1,1) double {mustBeInteger,mustBePositive}
    stream_offset (1,1) double {mustBeInteger,mustBeNonnegative} = 0
end
modulus = 2^31-2;
seed = mod(master_seed+104729*(repetition-1)+1009*stream_offset,modulus)+1;
end
