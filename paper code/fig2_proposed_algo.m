function fig2_proposed_algo
video = 0;
ploteach = 0;
%fig 1/(a)
% fig = 1;

%fig 1/(b) + fig 2/(a) constrained min = unconstrained min
% fig = 2;


%fig 1/(b) + fig 2/(b) constrained min ~= unconstrained min
% fig = 3;

%fig 1/(b) + fig 2/(c) constrained min ~= unconstrained min
% fig = 4;


global w_star; global b; global A; global a;global B;
%set up of parameters
dim = 2; b = 3; B = 20;A = 20; a = 0.2;

if 0 %calculate the constrained minimum
    N = 101; w1 = linspace(-1,1,N);obj_value = zeros(N,1);
    w_star = [0.5;0.2];
    for i = 1:N
%         w = [w1(i);w1(i)^2];
        w = [w1(i);sqrt(1-w1(i)^2)];
        obj_value(i) = obj_fcn_0(w);
    end
    plot(w1,obj_value)
%     w_star = [0.781475,sqrt(1-w1(i)^2)]; <-- the constrained min when w_star = [1/2,1/3] in Akley fcn
end




if video
    v = VideoWriter('constrained.mp4','MPEG-4');
    open(v);
end

%obj function and constrain fcn and accurate solution
if or(fig == 2, fig == 3) % x^2 + y^2 = 1
constrain_fcn = @(W)    sum(W.^2,1) - 1;
grad_constrain_fcn = @(W)   2*W;
Hess_constrain_fcn = @(W)   repmat(2*eye(size(W,1)),[1,1,size(W,2)]);
end

if fig == 4 % x^2 = y
constrain_fcn = @(W)    sum(W(1:end-1,:).^2,1) - W(end,:);
grad_constrain_fcn = @(W)   [2*W(1:end-1,:);-ones(1,size(W,2))];
Hess_constrain_fcn = @(W)   repmat(diag([ones(1,size(W,1)-1),0]),[1,1,size(W,2)]);
end


p_N = 50; p_batch = p_N; %N = number of particle; batch = number of particle to sum to get v
lambda = 1;  sigma = 1; alpha = 30; eps = 0.01; %parameter in equation
gama = 0.01; tN = 300; stop_criteria = 0; %parameter in leaning algorithm
lam = 1;


if fig == 1 %plot the obj function
    w_star = [0;0];

    N = 100; obj_value = zeros(N);
    x = linspace(-2,2,N);
    y = linspace(-2,2,N);
    for i = 1:N
            obj_value(i,:) = obj_fcn_0([x(i)*ones(1,N);y]);
    end
    mesh(y,x,obj_value);
    set(gca,'FontSize',30);
     set(gca,'xtick',[])
     set(gca,'ytick',[])
%     name = 'obj';
%     print(gcf, '-depsc', name);
end



if fig ~= 1
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

simu_N = 100; vW_success_rate = 0; dist = 0; %how many simulations


f_record = zeros(tN,simu_N);
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
        f(idx) = obj_fcn_0(W(:,idx))+lam *constrain_fcn(W).^2;

        %calculate v_W and fv
        omega = exp(-alpha*(f(idx)-min(f)));
        sum_omega = sum(omega);
        v_W = sum(W(:,idx) .* omega,2)/sum_omega;
        vW_record(:,t,simu) = v_W;
        f_record(t,simu) = obj_fcn_0(v_W);
        g_record(t,simu) = constrain_fcn(v_W);
        er_record(t,simu) = norm(v_W - v_star)/sqrt(dim);
        
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
        for j = 1:p_batch
            Hess_gsqr = 2*gd_g(:,j)*gd_g(:,j)' + 2*g(j)*Hess_g(:,:,j);
            temp = gama/eps*gd_gsqr(:,j) + temp_CBO(:,j);
            W(:,idx(j)) = W(:,idx(j)) -(eye(dim)+gama/eps*Hess_gsqr)\(temp);
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
semilogy(mean(er_record,2),'b-','LineWidth',2,'DisplayName','The Proposed algorithm');hold on;
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
% if fig == 3
% name = 'not_sphere';
% print(gcf, '-depsc', name);
% end


end
1;





    


