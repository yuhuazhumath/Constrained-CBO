function fig7
video = 0;
%obj function and constrain fcn and accurate solution
constrain_fcn = @(W)    sum(sum(W.^2,1) - 1,2);
constrain_fcn_sep = @(W)    sum(W.^2,1)-1;
grad_constrain_fcn = @(W)   2*W;
Hess_constrain_fcn = @(W)   2*repmat(eye(size(W,1)),[1,1,size(W,2)]);
mtv = @(W) reshape(W,[size(W,1)*size(W,2),size(W,3)]);
vtm = @(W) reshape(W,[3,round(size(W,1)/3),size(W,2)]);
N = 2; f_star = 0.5/N; %figure 7/a)
% N = 3; f_star = 1.732050808/N; %figure 7/b)
% N = 8; f_star =19.675287861/N; %figure 7/c)
% N = 15; f_star = 80.670244114/N; %figure 7/d)
% N = 56; f_star = 1337.094945276/N; %figure 7/e)
% N = 470; f_star = 104822.886324279/N; %figure 7/f)


%set up hyper parameters
p_N = 50;  %N = number of particle;
lambda = 1;  sigma = 1; alpha = 50; eps = 0.01; %parameter in equation
gama = 0.1; tN = 2000; stop_criteria = 10^(-14); %parameter in leaning algorithm
eps_min = 0.01; sig_indep = 0.3;



dim = (N-1)*3;
simu_N = 100; vW_success_rate = 0; dist = 0; 
f_record = zeros(tN,simu_N);
er_record = zeros(tN,simu_N);
g_record = zeros(tN,simu_N);
test1 = zeros(simu_N,1);
test2 = zeros(simu_N,1);
vW_final_record = zeros(dim,simu_N);
t_avg = 0;
for simu = 1:simu_N
    if video
        v = VideoWriter('constrained.mp4','MPEG-4');
        open(v);
    end

    %initialization of the weights
    f_min = 10^4;
    W1 = rand(1,N-1,p_N)*pi;
    W2 = rand(1,N-1,p_N)*2*pi;
    W = [sin(W1).*cos(W2);sin(W1).*sin(W2);cos(W1)];
    W = mtv(W);
    f = zeros(1,p_N);
    for t = 1:tN %this is for full batch
        
        idx = 1:p_N;

        %calculate f_j, for j \in B
        temp = obj_eg3(vtm(W(:,idx)));
        f(idx) = reshape(temp,[1,length(temp)]);
        
        
        %calculate v_W and fv
        omega = exp(-alpha*(f(idx) - min(f(idx))));
        sum_omega = sum(omega);
        v_W = sum(W(:,idx) .* omega,2)/sum_omega;
        f_record(t,simu) = obj_eg3(vtm(v_W));
        g_record(t,simu) = abs(constrain_fcn(vtm(v_W)));
        er_record(t,simu) = abs(f_record(t,simu) - f_star)/f_star;
        %vW_record(:,t,simu) = v_W;



         if video
            figure(6)
            plot(W(1,:),W(2,:),'b*'); 
            hold on;
            plot(v_W(1),v_W(2),'r*','MarkerSize',10)
            xlim([-3,3]); ylim([-3,3]);
            hold off;
            frame = getframe(gcf);
            writeVideo(v,frame);
         end







         %the step towards v_W
        diff = W(:,idx) - v_W;
        temp_CBO = lambda*gama*diff - sqrt(gama)*sigma*diff.*randn(size(diff));
         
        %update by algorithm 3 
        g_sep = constrain_fcn_sep(vtm(W(:,idx)));
        gd_g = grad_constrain_fcn(vtm(W(:,idx)));
        gd_gsqr = mtv(2*g_sep.*gd_g);
        Hess_g = Hess_constrain_fcn(W(:,idx));
        for j = 1:p_N
            Hess_gsqr = Hess_eg3(gd_g(:,:,j),g_sep(:,:,j),Hess_g(:,:,j));
            temp = gama/eps*gd_gsqr(:,j) + temp_CBO(:,j);
            W(:,idx(j)) = W(:,idx(j)) -(eye(dim)+gama/eps*Hess_gsqr)\(temp);
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

                figure(2)
                semilogy(er_record(1:t,simu),'Color',[.7,.7,1])
                hold on;
                
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
    test1(simu) = abs(obj_eg3(vtm(v_W)) - f_star)/f_star;
    test2(simu)= abs(constrain_fcn(vtm(v_W)));   
    
    if test1(simu) < 0.05 && test2(simu) < 0.001
        vW_success_rate = vW_success_rate+1;
    else
        1;
    end

    if simu == 55
        1;
    end

end

t_avg = t_avg/simu_N
[mean(test1), mean(test2), vW_success_rate]



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
figure(2); hold on;
semilogy(mean_er,'b-','LineWidth',2);
set(gca,'FontSize',30);
% name = sprintf('eq3_er_%d', N);
% print(gcf, '-depsc', name);




1;














    


