function [values, gradients] = ackley_objective(V, parameters)
%ACKLEY_OBJECTIVE Canonical column-wise Ackley objective from the manuscript.
arguments
    V double
    parameters struct
end

d = size(V, 1);
required = {'A', 'a', 'b', 'vhat'};
for i = 1:numel(required)
    assert(isfield(parameters, required{i}), ...
        'ackley_objective:MissingParameter', ...
        'Missing Ackley parameter "%s".', required{i});
end
assert(numel(parameters.vhat) == d, ...
    'ackley_objective:DimensionMismatch', ...
    'vhat must have one entry per row of V.');

delta = V - parameters.vhat(:);
radial = sqrt((parameters.b^2 / d) * sum(delta.^2, 1));
oscillatory = sum(cos(2*pi*parameters.b*delta), 1) / d;
values = -parameters.A * exp(-parameters.a * radial) ...
    - exp(oscillatory) + exp(1) + parameters.A;

if nargout>1
    radial_scale = zeros(size(radial));
    nonzero = radial>0;
    radial_scale(nonzero) = parameters.A*parameters.a ...
        *(parameters.b^2/d)*exp(-parameters.a*radial(nonzero)) ...
        ./radial(nonzero);
    gradients = delta.*radial_scale ...
        +(2*pi*parameters.b/d)*exp(oscillatory) ...
        .*sin(2*pi*parameters.b*delta);
end
end
