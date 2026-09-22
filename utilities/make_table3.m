function table3 = make_table3(input_files,output_file)
%MAKE_TABLE3 Thomson summary; error/feasibility means use successful runs only.
arguments
    input_files
    output_file char = ''
end
files=string(input_files(:)); n=numel(files);
case_id=strings(n,1); success_count=zeros(n,1); success_rate=zeros(n,1);
mean_relative_objective_error_success=NaN(n,1);
mean_l1_feasibility_success=NaN(n,1);
mean_feasibility_error_success=NaN(n,1);
mean_iterations=NaN(n,1);
for i=1:n
    [experiment,runs]=load_primary_runs(files(i)); case_id(i)=string(experiment.id);
    success=cellfun(@(x)x.success,runs); success_count(i)=sum(success); success_rate(i)=mean(success);
    if any(success)
        selected=runs(success);
        mean_relative_objective_error_success(i)=mean(cellfun(@(x)x.relative_objective_error,selected));
        mean_l1_feasibility_success(i)=mean(cellfun(@(x)x.l1_feasibility,selected));
        mean_feasibility_error_success(i)=mean(cellfun(@(x)x.feasibility_error,selected));
    end
    mean_iterations(i)=mean(cellfun(@(x)x.iterations,runs));
end
table3=table(case_id,success_count,success_rate, ...
    mean_relative_objective_error_success,mean_l1_feasibility_success, ...
    mean_feasibility_error_success,mean_iterations);
if ~isempty(output_file)
    parent=fileparts(output_file); if ~isempty(parent)&&~isfolder(parent), mkdir(parent); end
    writetable(table3,output_file);
end
end
