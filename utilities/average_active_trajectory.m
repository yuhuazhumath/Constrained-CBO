function mean_values = average_active_trajectory(values, active_mask)
%AVERAGE_ACTIVE_TRAJECTORY Average active entries without treating zero as missing.
arguments
    values double
    active_mask logical
end
assert(isequal(size(values),size(active_mask)), ...
    'average_active_trajectory:SizeMismatch', ...
    'values and active_mask must have identical size.');
masked = values;
masked(~active_mask) = NaN;
mean_values = mean(masked,2,'omitnan');
end
