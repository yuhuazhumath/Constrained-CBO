function table1 = make_table1(input_files,output_file)
%MAKE_TABLE1 Long-form Family-A summary generated from raw results.
arguments
    input_files
    output_file char = ''
end
table1 = make_summary_table(input_files,output_file);
end
