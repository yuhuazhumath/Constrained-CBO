function fig2_Penalty_CBO
video = 0;
ploteach = 0;

%fig 1/(b) + fig 2/(a) constrained min = unconstrained min
if (1)
    fig = 2;
    type = 1;
end

%fig 1/(b) + fig 2/(b) constrained min ~= unconstrained min
if (0)
    fig = 3;
    type = 1;
end

%fig 1/(b) + fig 2/(c) constrained min ~= unconstrained min
if (0)
    fig = 4;
    type = 2;
end

global w_star; global b; global A; global a;global B;global eps;
%set up of parameters
dim = 2; b = 3; B = 20;A = 20; a = 0.2;

if video
    v = VideoWriter('constrained.mp4','MPEG-4');
    open(v);
end

%obj function and constrain fcn and accurate solution
constrain_fcn = @(W)    sum(W.^2,1) - 1;
grad_constrain_fcn = @(W)   2*W;
Hess_constrain_fcn = @(W)   repmat(2*eye(size(W,1)),[1,1,size(W,2)]);


p_N = 50; p_batch = p_N; %N = number of particle; batch = number of particle to sum to get v
lambda = 1;  sigma = 1; alpha = 30; eps = 10^(-2); %parameter in equation
gama = 0.01; tN = 300; stop_criteria = 0; %parameter in leaning algorithm


if fig == 3 %when unconstrained  != constrained
    w_star = [1/2;1/3]; 
    v_star = [0.781475;sqrt(1-0.781475^2)];
end


if fig == 2%when unconstrained  = constrained
    w_star = [1/sqrt(2);-1/sqrt(2)]; 
    v_star = w_star;
end

if fig == 4
    w_star = [1/2;1/3]; 
    v_star = [0.5428;0.5428^2];
end

if (0)
    N = 100; obj_value = zeros(N);
    x = linspace(-2,2,N);
    y = linspace(-2,2,N);
    for i = 1:N
            obj_value(i,:) = obj_fcn_penalty([x(i)*ones(1,N);y],type);
    end
    mesh(y,x,obj_value);
    set(gca,'FontSize',30);
     set(gca,'xtick',[])
     set(gca,'ytick',[])
    name = 'obj_penalized';
    print(gcf, '-depsc', name);
end

simu_N = 100; vW_success_rate = 0; dist = 0; %how many simulations


f_record = zeros(tN,simu_N);
f_penalty_record = zeros(tN,simu_N);
g_record = zeros(tN,simu_N);
vW_record = zeros(dim, tN,simu_N);
vW_final_record = zeros(dim,simu_N);
er_record = zeros(tN,simu_N);
for simu = 1:simu_N
    %initialization of the weights
    W = rand(dim,p_N)*6-3;
    f = zeros(1,p_N);
    for t = 1:tN %this is for full batch
        idx = 1:p_N;

        %calculate f_j, for j \in B
        f(idx) = obj_fcn_penalty(W(:,idx),type);

        %calculate v_W and fv
        omega = exp(-alpha*(f(idx)-min(f)));
        sum_omega = sum(omega);
        v_W = sum(W(:,idx) .* omega,2)/sum_omega;
        vW_record(:,t,simu) = v_W;
        f_penalty_record(t,simu) = obj_fcn_penalty(v_W,type);
        g_record(t,simu) = constrain_fcn(v_W);
        er_record(t,simu) = norm(v_W - v_star)/sqrt(dim);
        
        %the step towards v_W
        diff = W(:,idx) - v_W;
        temp_CBO = lambda*gama*diff - sqrt(gama)*sigma*diff.*randn(size(diff));
         
        %update by algorithm 3 
        W(:,idx) = W(:,idx) - temp_CBO;


        if video
            figure(1)
            plot(W(1,:),W(2,:),'b*'); 
            hold on;
            plot(v_W(1),v_W(2),'r*')
            plot(w_star(1),w_star(2),'k*')
            plot(v_star(1),v_star(2),'g*')
            xlim([-3,3]); ylim([-3,3]);
            x1 = linspace(-1,1,100); x2 = real(sqrt(1-x1.^2));
            plot(x1,x2,'k'); hold on; plot(x1,-x2,'k');xlim([-3,3]); ylim([-3,3]);
            frame = getframe(gcf);
            writeVideo(v,frame);
            hold off;
        end
        
        
        %check step criteria
        concentrate = sum(sum((W(:,idx) - v_W).^2))/dim/p_N;
        if or(concentrate < stop_criteria, t == tN)
            if ploteach == 1
                figure(4)
                semilogy(er_record(1:t,simu),'Color',[.7,.7,1])
                hold on;
            end
            break;
        end



    end
    if video
        close(v);
    end
    
    vW_final_record(:,simu) = v_W;
    test = abs(v_W - v_star) < 0.01;
    if sum(test) == dim
        vW_success_rate = vW_success_rate+1;
    end
    dist = dist + norm(v_W - v_star,2)/sqrt(dim);
end
dist = dist/simu_N;
[dist,vW_success_rate]

figure(4)
semilogy(mean(er_record,2),'m-','LineWidth',2,'DisplayName','Penalized CBO');
xlim([1,300]); 
set(gca,'FontSize',30);
% if fig == 2
% name = 'UnConst_is_const';
% print(gcf, '-depsc', name);
% end
% if fig == 3
% name = 'UnConst_isnot_const';
% print(gcf, '-depsc', name);
% end
% 
% if fig == 4
% name = 'not_sphere';
% print(gcf, '-depsc', name);
% end

1;





    


