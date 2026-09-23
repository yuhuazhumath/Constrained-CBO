function problem = thomson_problem(k)
%THOMSON_PROBLEM Symmetry-reduced k-electron manuscript problem.
% Electron 1 is fixed at (1,0,0); only electrons 2,...,k are variables.
arguments
    k (1,1) double {mustBeInteger, mustBeGreaterThanOrEqual(k,2)}
end
reference_total = containers.Map( ...
    {2,3,8,15,56,470}, ...
    {0.5,1.732050808,19.675287861,80.670244114,1337.094945276,104822.886324279});
assert(isKey(reference_total,k), 'thomson_problem:MissingReference', ...
    'No manuscript reference energy is available for k=%d.', k);
fixed_electron = [1;0;0];
free_electrons = k-1;
d = 3*free_electrons;
problem = struct();
problem.name = sprintf('thomson_k%d',k);
problem.dimension = d;
problem.num_constraints = free_electrons;
problem.nominal_electron_count = k;
problem.num_free_electrons = free_electrons;
problem.fixed_electron = fixed_electron;
problem.reconstruct_full = @(V) reconstruct_full_configuration( ...
    V,fixed_electron,free_electrons);
problem.E = @(V) thomson_energy(V,k,fixed_electron,free_electrons);
problem.g = @(V) thomson_constraints(V,free_electrons);
problem.G = @(V) sum(thomson_constraints(V,free_electrons).^2,1);
problem.gradG = @(V) thomson_gradG(V,free_electrons);
problem.hessG = @(V) thomson_hessG(V,free_electrons);
problem.solve_implicit_matrix = @(V,rhs,coefficient) ...
    thomson_implicit_solve(V,rhs,coefficient,free_electrons);
problem.vstar = [];
problem.objective_star = reference_total(k)/k;
problem.l1_feasibility = @(V) sum(abs(thomson_constraints(V,free_electrons)),1);
problem.metadata = struct('objective','normalized Thomson energy', ...
    'constraint','k-1 free-electron unit-sphere equalities', ...
    'constraint_representation','vector g; G=||g||_2^2', ...
    'k',k,'nominal_electron_count',k, ...
    'num_free_electrons',free_electrons, ...
    'optimization_dimension',d,'ambient_physical_dimension',3*k, ...
    'fixed_electron',fixed_electron, ...
    'initialization','theta~Unif(0,pi), phi~Unif(0,2*pi)', ...
    'reference_total_energy',reference_total(k), ...
    'reference_normalization','total pairwise energy divided by nominal k');
end

function values = thomson_energy(V,k,fixed_electron,free_electrons)
n = size(V,2);
X = reconstruct_full_configuration(V,fixed_electron,free_electrons);
values = zeros(1,n);
upper_triangle = triu(true(k),1);
for p = 1:n
    positions = X(:,:,p);
    squared_norms = sum(positions.^2,1);
    squared_distances = max(squared_norms.'+squared_norms ...
        -2*(positions.'*positions),0);
    values(p) = sum(1./sqrt(squared_distances(upper_triangle)))/k;
end
end

function full = reconstruct_full_configuration(V,fixed_electron,free_electrons)
n = size(V,2);
free = reshape(V,3,free_electrons,n);
full = cat(2,repmat(fixed_electron,1,1,n),free);
end

function solution = thomson_implicit_solve(V,rhs,coefficient,free_electrons)
% Solve the independent 3-by-3 electron blocks without a full Hessian.
n = size(V,2);
X = reshape(V,3,free_electrons,n);
B = reshape(rhs,3,free_electrons,n);
solution_blocks = zeros(size(B));
identity3 = eye(3);
for p = 1:n
    for i = 1:free_electrons
        x = X(:,i,p);
        constraint = x.'*x-1;
        block = (1+4*coefficient*constraint)*identity3 ...
            +8*coefficient*(x*x.');
        solution_blocks(:,i,p) = block\B(:,i,p);
    end
end
solution = reshape(solution_blocks,3*free_electrons,n);
end

function g = thomson_constraints(V,free_electrons)
X = reshape(V,3,free_electrons,size(V,2));
g = reshape(sum(X.^2,1)-1,free_electrons,size(V,2));
end

function grad = thomson_gradG(V,free_electrons)
n = size(V,2);
X = reshape(V,3,free_electrons,n);
g = reshape(sum(X.^2,1)-1,1,free_electrons,n);
grad = reshape(4*g.*X,3*free_electrons,n);
end

function H = thomson_hessG(V,free_electrons)
assert(size(V,2)==1,'thomson_problem:HessianSinglePoint', ...
    ['Thomson hessG returns one sparse analytical Hessian at a time; ', ...
    'the batched Equation (33) update uses solve_implicit_matrix.']);
X = reshape(V,3,free_electrons);
g = sum(X.^2,1)-1;
row_indices = zeros(9*free_electrons,1);
column_indices = zeros(9*free_electrons,1);
entries = zeros(9*free_electrons,1);
for i = 1:free_electrons
    rows = (3*i-2):(3*i);
    x = X(:,i);
    block = 8*(x*x.') + 4*g(i)*eye(3);
    indices = (9*i-8):(9*i);
    [block_rows,block_columns] = ndgrid(rows,rows);
    row_indices(indices) = block_rows(:);
    column_indices(indices) = block_columns(:);
    entries(indices) = block(:);
end
dimension = 3*free_electrons;
H = sparse(row_indices,column_indices,entries,dimension,dimension, ...
    9*free_electrons);
end
