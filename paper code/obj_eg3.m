function temp = obj_eg3(W)
temp = sum(ones(size(W(1,:,:)))./sqrt(sum((W - [1;0;0]).^2,1)),2);
for i = 2:size(W,2)
    temp = temp + sum( ones(size(W(1,1:i-1,:)))...
        ./sqrt(sum((W(:,1:i-1,:) - W(:,i,:)).^2,1)), 2);
end
temp = temp/(size(W,2)+1);
    