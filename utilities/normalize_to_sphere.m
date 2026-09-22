function V = normalize_to_sphere(base_samples)
%NORMALIZE_TO_SPHERE Normalize columns to S^(d-1), mapping zero columns to e1.
norms = sqrt(sum(base_samples.^2,1));
zero = norms==0;
if any(zero)
    base_samples(:,zero) = 0;
    base_samples(1,zero) = 1;
    norms(zero) = 1;
end
V = base_samples./norms;
end
