function save_revision_checkpoint(output_file,experiment)
%SAVE_REVISION_CHECKPOINT Atomically replace a revision-experiment MAT file.
checkpoint_file = [output_file,'.partial'];
save(checkpoint_file,'experiment','-v7.3');
[moved,message] = movefile(checkpoint_file,output_file,'f');
assert(moved,'save_revision_checkpoint:MoveFailed', ...
    'Could not install checkpoint %s: %s',output_file,message);
end
