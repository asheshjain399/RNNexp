close all;
clc;

addpaths;
%erd = '../checkpoints_malik_T_150_bs_100_tg_100_initial_lr_0.001_clipnorm_25.0_nschd_[250,500.0,1000.0,1300.0,2000.0,2500.0,4000.0]_nrate_[0.01,0.05,0.1,0.2,0.3,0.5,0.65]_decschd_[1500.0,4500.0]_decrate_[0.1,0.1]_fs_final';
%dra = '../checkpoints_dra_T_150_bs_100_tg_100_ls_512_fc_256_initial_lr_0.001_clipnorm_25.0_nschd_[250,500.0,1000.0,1300.0,2000.0,2500.0,3300.0]_nrate_[0.01,0.05,0.1,0.2,0.3,0.5,0.68]_decschd_[1500.0,4500.0]_decrate_[0.1,0.1]_fs_final';

erd = '../checkpoints_malik_T_150_bs_100_tg_100_initial_lr_0.001_clipnorm_25.0_nschd_]_nrate_]_decschd_[1500.0,4500.0]_decrate_[0.1,0.1]_fs_final';
dra = 'checkpoints_dra_T_150_bs_100_tg_100_ls_512_fc_256_initial_lr_0.001_clipnorm_25.0_nschd_[250,500.0,1000.0,1300.0,2000.0,2500.0,3300.0]_nrate_[0.01,0.05,0.1,0.2,0.3,0.5,0.7]_decschd_[1500.0,4500.0]_decrate_[0.1,0.1]_fs_final';

iterations = 3500:250:5500;
cc = 1;
for i = 500:250:5500
    mean_error = motionGenerationError( dra,i );
    if size(mean_error,1) > 0
        dra_errors(cc) = mean_error(1);
        cc = cc + 1;
    else
        break;
    end;
    close all;
    
end;

[loss,validation] = readLogFile(erd);
loss = sqrt(loss*1.0/T)*(1.0/54);
loss = interp1(1:numel(loss),loss,1:numel(loss));
validation = validation*1.0/T;
valerr = (sqrt((validation(find(validation>0)))))*1.0/54;
p_erd = plot(loss,'color','r','linewidth',3);
hold on;
p_erd_test = plot(find(validation>0),valerr,'color','r','linewidth',3,'linestyle','--');
hold on;

[loss,validation] = readLogFile(dra);
loss = sqrt(loss*1.0/T)*(1.0/54);
loss = interp1(1:numel(loss),loss,1:numel(loss));
validation = validation*1.0/T;
valerr = (sqrt((validation(find(validation>0)))))*1.0/54;
p_dra = plot(loss,'color','b','linewidth',3);
hold on;
p_dra_test = plot(find(validation>0),valerr,'color','b','linewidth',3,'linestyle','--');

l = legend([p_erd,p_erd_test,p_dra,p_dra_test],{'ERD Train Error','ERD Test Error','S-RNN Train Error','S-RNN Test Error'},'fontsize',20);

%xlabel('Iterations','fontsize',16);
%ylabel('Error','fontsize',16);
ax = gca;
set(ax,'fontsize',24);
xlim([0,4000]);
print('lossPlot','-dpng','-r600');
close all;
%print('-depsc','lossPlot.eps');
%unix(['epstopdf lossPlot.eps']);