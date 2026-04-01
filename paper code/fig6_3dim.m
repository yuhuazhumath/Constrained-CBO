function fig6_3dim
%3-dimensional case
const = 1;%figure 6/a)
% const = 2;%figure 6/b)
% const = 3;%figure 6/c)
video = 0;


%obj function and constrain fcn and accurate solution
if const == 1
constrain_fcn = @(W)    sum(W.^2,1) - 1;
grad_constrain_fcn = @(W)   2*W;
Hess_constrain_fcn = @(W)   repmat(eye(size(W,1))*2,[1,1,size(W,2)]);
end

if const == 2
constrain_fcn = @(W)    sum(W(1:end-1,:).^2,1) - W(end,:);
grad_constrain_fcn = @(W)   [2*W(1:end-1,:);-ones(1,size(W,2))];
Hess_constrain_fcn = @(W)   repmat(diag([ones(1,size(W,1)-1),0]),[1,1,size(W,2)]);
end

if const == 3
constrain_fcn = @(W)    sum(W(1:end,:),1) - 1;
grad_constrain_fcn = @(W)   ones(size(W));
Hess_constrain_fcn = @(W)   repmat(zeros(size(W,1)),[1,1,size(W,2)]);

constrain_fcn2 = @(W)    2*sum(W(1:end-1,:),1)-1/2*W(end,:)-1/2;
grad_constrain_fcn2 = @(W)   [2*ones(size(W,1)-1,size(W,2));-1/2*ones(1,size(W,2))];
Hess_constrain_fcn2 = @(W)   repmat(zeros(size(W,1)),[1,1,size(W,2)]);
end



%set up hyper parameters
p_N = 100;  %N = number of particle;
lambda = 1;  sigma = 1; alpha = 50; eps = 0.01; %parameter in equation
gama = 0.1; tN = 1000; stop_criteria = 10^(-14); %parameter in leaning algorithm

dim = 3; 
if const == 1
w_star = [sqrt(1/3);sqrt(1/3);sqrt(1/3)];
end
if const == 2
w_star = [0.4283;0.4283;0.3669];
end
if const == 3
w_star = [0.2;0.2;0.6];
end


simu_N = 100; vW_success_rate = 0; dist = 0; 


er_record = zeros(tN,simu_N);
vW_final_record = zeros(dim,simu_N);

if video
    v = VideoWriter('constrained.mp4','MPEG-4');
    open(v);
end
t_avg = 0;
for simu = 1:simu_N
    %initialization of the weights
    W = rand(dim,p_N)*6-3;
    f = zeros(1,p_N);
    for t = 1:tN %this is for full batch
        
        idx = 1:p_N;

        %calculate f_j, for j \in B
        f(idx) = obj_fcn(W(:,idx));

        %calculate v_W and fv
        omega = exp(-alpha*(f(idx)-min(f)));
        sum_omega = sum(omega);
        v_W = sum(W(:,idx) .* omega,2)/sum_omega;
        er_record(t,simu) = norm(v_W - w_star)/sqrt(dim);


         if video
            figure(3)
            plot(W(1,:),W(2,:),'b*'); 
            hold on;
            plot(v_W(1),v_W(2),'r*','MarkerSize',10)
            plot(w_star(1),w_star(2),'g*','MarkerSize',10)
            plot(0.4,0.4,'k*','MarkerSize',10)
            xlim([-4,4]); ylim([-4,4]);
            hold off;
            frame = getframe(gcf);
            writeVideo(v,frame);
        end
        
        %the step towards v_W
        diff = W(:,idx) - v_W;
        temp_CBO = lambda*gama*diff - sqrt(gama)*sigma*diff.*randn(size(diff));
         
        %update by algorithm 3 
        g = constrain_fcn(W(:,idx));
        gd_g = grad_constrain_fcn(W(:,idx));
        Hess_g = Hess_constrain_fcn(W(:,idx));
        gd_gsqr = 2*g.*gd_g;
        if const == 3
            g2 = constrain_fcn2(W(:,idx));
            gd_g2 = grad_constrain_fcn2(W(:,idx));
            Hess_g2 = Hess_constrain_fcn2(W(:,idx));
            gd_gsqr2 = 2*g2.*gd_g2;
        end
        for j = 1:p_N
            Hess_gsqr = 2*gd_g(:,j)*gd_g(:,j)' + 2*g(j)*Hess_g(:,:,j);
            temp = gama/eps*gd_gsqr(:,j) + temp_CBO(:,j);
            if const == 3
                Hess_gsqr2 = 2*gd_g2(:,j)*gd_g2(:,j)' + 2*g2(j)*Hess_g2(:,:,j);
                Hess_gsqr = Hess_gsqr + Hess_gsqr2;
                temp = temp + gama/eps*gd_gsqr2(:,j);
            end
            W(:,idx(j)) = W(:,idx(j)) -(eye(dim)+gama/eps*Hess_gsqr)\temp;
        end

        
        
        %check step criteria
        concentrate = sum(sum((W(:,idx) - v_W).^2))/dim/p_N;
        if or(concentrate < stop_criteria, t == tN)
            figure(2)
            semilogy(er_record(1:t,simu),'Color',[.7,.7,1])
            hold on;
            
            omega = exp(-alpha*(f(idx)-min(f)));
            sum_omega = sum(omega);
            vW_final_record(:,simu) = sum(W(:,idx) .* omega,2)/sum_omega;
            t_avg = t_avg + t;
            break;
        end



    end
    if video
        close(v);
    end
    
    test = abs(v_W - w_star) < 0.1;
    if sum(test) == dim
        vW_success_rate = vW_success_rate+1;
    end
    dist = dist + norm(v_W - w_star)/sqrt(dim);
end
dist = dist/simu_N;
t_avg = t_avg/simu_N
[dist,vW_success_rate]

figure(2)
mean_er = zeros(tN,1);
for t = 1:tN
    idx = er_record(t,:) == 0;
    if sum(idx) == simu_N
        break;
    else
        mean_er(t) =  sum(er_record(t,:))/(simu_N-sum(idx));
    end
end
mean_er = mean_er(1:t-1);
semilogy(mean_er,'b-','LineWidth',2);
set(gca,'FontSize',30);
% name = 'eq2_dim3_er_3';
% print(gcf, '-depsc', name);

1;


