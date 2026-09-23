function metrics = compute_metrics(problem, v_out, config)
%COMPUTE_METRICS Universal final metrics evaluated at one v_out.
arguments
    problem struct
    v_out double
    config struct
end
metrics = struct();
metrics.v_out = v_out;
metrics.objective_value = problem.E(v_out);
metrics.G_value = problem.G(v_out);
metrics.feasibility_error = sqrt(max(metrics.G_value,0));
if isempty(problem.vstar)
    metrics.distance_to_vstar = NaN;
    metrics.infinity_distance_to_vstar = NaN;
else
    metrics.distance_to_vstar = norm(v_out-problem.vstar)/sqrt(problem.dimension);
    metrics.infinity_distance_to_vstar = norm(v_out-problem.vstar,Inf);
end
if isfield(problem,'objective_star') && isfinite(problem.objective_star)
    metrics.objective_error = abs(metrics.objective_value-problem.objective_star);
    if problem.objective_star~=0
        metrics.relative_objective_error = metrics.objective_error/abs(problem.objective_star);
    else
        metrics.relative_objective_error = NaN;
    end
else
    metrics.objective_error = NaN;
    metrics.relative_objective_error = NaN;
end
if isfield(problem,'l1_feasibility')
    metrics.l1_feasibility = problem.l1_feasibility(v_out);
elseif isfield(problem,'g')
    metrics.l1_feasibility = sum(abs(problem.g(v_out)),1);
else
    metrics.l1_feasibility = NaN;
end

checks = true;
if ~isempty(config.success_distance_tol)
    checks = checks && isfinite(metrics.distance_to_vstar) ...
        && metrics.distance_to_vstar<=config.success_distance_tol;
end
if ~isempty(config.success_infinity_distance_tol)
    checks = checks && isfinite(metrics.infinity_distance_to_vstar) ...
        && metrics.infinity_distance_to_vstar<=config.success_infinity_distance_tol;
end
if ~isempty(config.success_feasibility_tol)
    checks = checks && metrics.feasibility_error<=config.success_feasibility_tol;
end
if ~isempty(config.success_relative_objective_tol)
    checks = checks && metrics.relative_objective_error ...
        <=config.success_relative_objective_tol;
end
if ~isempty(config.success_l1_feasibility_tol)
    checks = checks && metrics.l1_feasibility<=config.success_l1_feasibility_tol;
end
metrics.success = logical(checks);
end
