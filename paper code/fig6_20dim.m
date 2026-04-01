function fig6_20dim
%20 dimensional case
% const = 1; %figure 6/d)
const = 2; %figure 6/e)
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
gama = 0.1; tN = 5*10^3; stop_criteria = 10^(-5); %parameter in leaning algorithm

if const == 1
    eps_min = 0.01;sig_indep = 0.3;
end
if const == 2
    eps_min = 0.001; sig_indep = 1;
end
if const == 3
    eps_min = 0.01; sig_indep = 0.3;
end


dim =20;

if const == 1
w_star = sqrt(1/dim)*ones(dim,1);
end
if const == 2
w_star = [0.3542*ones(dim-1,1);2.3839]; 
end 


simu_N = 100; vW_success_rate = 0; dist = 0; 


f_record = zeros(tN,simu_N);
er_record = zeros(tN,simu_N);
g_record = zeros(tN,simu_N);
vW_final_record = zeros(dim,simu_N);
t_avg = 0;

for simu = 1:simu_N
    if video
        v = VideoWriter('constrained.mp4','MPEG-4');
        open(v);
    end

    %initialization of the weights
    f_min = 10^4;
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
        f_record(t,simu) = obj_fcn(v_W);
        g_record(t,simu) = constrain_fcn(v_W);
        if const == 3
        g_record(t,simu) = g_record(t,simu) + constrain_fcn2(v_W);
        end
        er_record(t,simu) = norm(v_W - w_star)/sqrt(dim);


         if video
            figure(4)
            plot(W(1,:),W(2,:),'b*'); 
            hold on;
            plot(v_W(1),v_W(2),'r*','MarkerSize',10)
            plot(w_star(1),w_star(2),'g*','MarkerSize',10)
            plot(1/2,1/2,'k*','MarkerSize',10)
            xlim([-4,4]); ylim([-4,4]);
            %x1 = linspace(-4,4,100); x2 = 3-x1;
            %plot(x1,x2,'k');
            hold off;
            frame = getframe(gcf);
            writeVideo(v,frame);
        end
        
        %the step towards v_W
        diff = W(:,idx) - v_W;
        temp_CBO = lambda*gama*diff - sqrt(gama)*sigma*diff.*randn(size(diff));
         
        %update by algorithm
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

        %check stop criteria
        concentrate = sum(sum((W(:,idx) - v_W).^2))/dim/p_N; 
        
        if or(concentrate < stop_criteria, t == tN)
            if and(abs(f_min - f_record(t,simu))>eps_min, t<tN)
                if f_record(t,simu) < f_min
                    f_min = f_record(t,simu);
                    vW_final_record(:,simu) = v_W;
                end
                W(:,idx) = W(:,idx) + sig_indep*sqrt(gama)*randn(size(W(:,idx)));
            else
                
                
                if const ~= 3
                figure(2)
                semilogy(er_record(1:t,simu),'Color',[.7,.7,1])
                hold on;
                end

                if f_min > f_record(t,simu)
                    vW_final_record(:,simu) = v_W;
                end
                t_avg = t_avg + t;
                break;
            end
        end



    end
    if video
        close(v);
    end

    v_W = vW_final_record(:,simu);
    test = abs(v_W - w_star) < 0.1;
    if sum(test) == dim
        vW_success_rate = vW_success_rate+1;
    end
    dist = dist + norm(v_W - w_star)/sqrt(dim);
end
dist = dist/simu_N;
t_avg = t_avg/simu_N
[dist,vW_success_rate]


mean_er = zeros(tN,1);
mean_g = zeros(tN,1);
mean_f = zeros(tN,1);
for t = 1:tN
    idx = er_record(t,:) == 0;
    if sum(idx) == simu_N
        break;
    else
        mean_er(t) =  sum(er_record(t,:))/(simu_N-sum(idx));
        mean_f(t) =  sum(f_record(t,:))/(simu_N-sum(idx));
        mean_g(t) =  sum(g_record(t,:))/(simu_N-sum(idx));
    end
end
mean_er = mean_er(1:t-1);
mean_f = mean_f(1:t-1);
mean_g = mean_g(1:t-1);

if const ~= 3
    figure(2)
    hold on;
    semilogy(mean_er,'b-','LineWidth',2);
    set(gca,'FontSize',30);
    name = 'eq2_dim20_er_2';
    print(gcf, '-depsc', name);
end


1;


