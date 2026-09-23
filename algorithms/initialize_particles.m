function V = initialize_particles(problem, config)
%INITIALIZE_PARTICLES Sample particles from the active RNG stream.
if ~isempty(config.initial_particles)
    V = config.initial_particles;
    return
end
kind = config.initialization.type;
switch kind
    case 'uniform_box'
        lo = config.initialization.lower;
        hi = config.initialization.upper;
        V = lo+(hi-lo)*rand(problem.dimension,config.particles);
    case 'normal'
        V = randn(problem.dimension,config.particles);
    case 'thomson_angular'
        assert(isfield(problem,'num_free_electrons') ...
            && problem.dimension==3*problem.num_free_electrons, ...
            'initialize_particles:InvalidThomsonProblem', ...
            'Angular initialization requires a reduced Thomson problem.');
        % Uniform polar and azimuthal angles, rather than uniform surface area.
        theta = pi*rand(1,problem.num_free_electrons,config.particles);
        phi = 2*pi*rand(1,problem.num_free_electrons,config.particles);
        free = [sin(theta).*cos(phi); ...
                sin(theta).*sin(phi); ...
                cos(theta)];
        V = reshape(free,problem.dimension,config.particles);
    otherwise
        error('initialize_particles:UnknownType', ...
            'Unknown initialization type "%s".',kind);
end
end
