function fig4_linear
%plot figure 4 / c) & d)
%the linear constraint
video = 0;
if video
    v = VideoWriter('constrained.mp4','MPEG-4');
    open(v);
end

%obj function and constrain fcn and accurate solution
obj_fcn = @(W)      sum(W.^2,1);
constrain_fcn = @(W)    sum(W,1) - 3;
grad_constrain_fcn = @(W)   W;
Hess_constrain_fcn = @(W)   repmat([0,0;0,0],[1,1,size(W,2)]);


%set up hyper parameters
p_N = 50;  %N = number of particle;
lambda = 1;  sigma = 1; alpha = 50; eps = 0.01; %parameter in equation
gama = 0.1; tN = 100; stop_criteria = 10^(-14); %parameter in leaning algorithm

%problem set up
dim = 2;  w_star = [3/2;3/2];
simu_N = 100; vW_success_rate = 0; dist = 0; 


f_record = zeros(tN,simu_N);
g_record = zeros(tN,simu_N);
er_record = zeros(tN,simu_N);
vW_record = zeros(dim, tN,simu_N);
vW_final_record = zeros(dim,simu_N);


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
        vW_record(:,t,simu) = v_W;
        f_record(t,simu) = obj_fcn(v_W);
        g_record(t,simu) = constrain_fcn(v_W);
        er_record(t,simu) = norm(v_W - w_star)/sqrt(dim);

        if video
            figure(3)
            plot(0,0,'k*','MarkerSize',10,'LineWidth',1);
            hold on;
            plot(W(1,:),W(2,:),'b*'); 
            plot(w_star(1),w_star(2),'g*','MarkerSize',10,'LineWidth',2)
            plot(v_W(1),v_W(2),'r*','MarkerSize',10,'LineWidth',2)
            xlim([-4,4]); ylim([-4,4]);
            x1 = linspace(-4,4,100); x2 = 3-x1;
            plot(x1,x2,'k');
            hold off;
            frame = getframe(gcf);
            writeVideo(v,frame);

         if or(or(or ( t==5, t==1), t==50), t==100)
             set(gca,'FontSize',30);
             1;
         end
        end

        %the step towards v_W
        diff = W(:,idx) - v_W;
        temp_CBO = lambda*gama*diff - sqrt(gama)*sigma*diff.*randn(size(diff));
         
        %update by algorithm 3 
        g = constrain_fcn(W(:,idx));
        gd_g = grad_constrain_fcn(W(:,idx));
        Hess_g = Hess_constrain_fcn(W(:,idx));
        gd_gsqr = 2*g.*gd_g;
        for j = 1:p_N
            Hess_gsqr = 2*gd_g(:,j)*gd_g(:,j)' + 2*g(j)*Hess_g(:,:,j);
            temp = gama/eps*gd_gsqr(:,j) + temp_CBO(:,j);
            W(:,idx(j)) = W(:,idx(j)) -(eye(dim)+gama/eps*Hess_gsqr)\temp;
        end


        
        
        %check step criteria
        concentrate = sum(sum((W(:,idx) - v_W).^2))/dim/p_N;
        if or(concentrate < stop_criteria, t == tN)
            figure(1)
            if simu == 1
            plot([1,t],[obj_fcn(w_star), obj_fcn(w_star)], 'b--','LineWidth',2);
            hold on;
            plot([1,t],[constrain_fcn(w_star), constrain_fcn(w_star)], 'r--','LineWidth',2);
            hold on;
            end
            plot(f_record(1:t,simu),'Color',[.7,.7,1]);
            plot(g_record(1:t,simu),'Color',[1,.7,.7]);

            figure(2)
            semilogy(er_record(1:t,simu),'Color',[.7,.7,1])
            hold on;
            
            omega = exp(-alpha*(f(idx)-min(f)));
            sum_omega = sum(omega);
            vW_final_record(:,simu) = sum(W(:,idx) .* omega,2)/sum_omega;
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
[dist,vW_success_rate]

figure(1)
plot(mean(f_record,2),'b-','LineWidth',2);
plot(mean(g_record,2),'r-','LineWidth',2);
xlim([1,100]); ylim([-2,6]);
set(gca,'FontSize',30);
% name = 'eq1_1_plane';
% print(gcf, '-depsc', name);

figure(2)
semilogy(mean(er_record,2),'b-','LineWidth',2);
set(gca,'FontSize',30);
% name = 'eq1_2_plane';
% print(gcf, '-depsc', name);

1;


