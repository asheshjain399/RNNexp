filenames = {};
filenames{1,1} = 'checkpoints_dra_T_150_bs_100_tg_100_ls_512_fc_256_initial_lr_0.001_clipnorm_25.0_nschd_[250,500.0,1000.0,1300.0,2000.0,2500.0,3300.0]_nrate_[0.01,0.05,0.1,0.2,0.3,0.5,0.7]_decschd_[1500.0,4500.0]_decrate_[0.1,0.1]_fs_final';
filenames{2,1} = 'checkpoints_dra_T_150_bs_100_tg_100_ls_512_fc_256_initial_lr_0.001_clipnorm_25.0_nschd_[250,500.0,1000.0,1300.0,2000.0,2500.0,3300.0]_nrate_[0.01,0.05,0.1,0.2,0.3,0.5,0.68]_decschd_[1500.0,4500.0]_decrate_[0.1,0.1]_fs_smoking';
filenames{3,1} = 'checkpoints_dra_T_150_bs_100_tg_100_ls_512_fc_256_initial_lr_0.001_clipnorm_25.0_nschd_[250,500.0,1000.0,1300.0,2000.0,2500.0,3300.0]_nrate_[0.01,0.05,0.1,0.2,0.3,0.5,0.68]_decschd_[1500.0,4500.0]_decrate_[0.1,0.1]_fs_eating';
filenames{4,1} = 'checkpoints_dra_T_150_bs_100_tg_100_ls_512_fc_256_initial_lr_0.001_clipnorm_25.0_nschd_[250,500.0,1000.0,1300.0,2000.0,2500.0,3300.0]_nrate_[0.01,0.05,0.1,0.2,0.3,0.5,0.7]_decschd_[1500.0,4500.0]_decrate_[0.1,0.1]_fs_discussion';

filenames{5,1} = 'checkpoints_malik_T_150_bs_100_tg_100_initial_lr_0.001_clipnorm_25.0_nschd_[250,500.0,1000.0,1300.0,2000.0,2500.0,4000.0]_nrate_[0.01,0.05,0.1,0.2,0.3,0.5,0.65]_decschd_[1500.0,4500.0]_decrate_[0.1,0.1]_fs_final';
filenames{6,1} = 'checkpoints_malik_T_150_bs_100_tg_100_initial_lr_0.001_clipnorm_25.0_nschd_[250,500.0,1000.0,1300.0,2000.0,2500.0,3300.0]_nrate_[0.01,0.05,0.1,0.2,0.3,0.5,0.68]_decschd_[1500.0,4500.0]_decrate_[0.1,0.1]_fs_smoking';
filenames{7,1} = 'checkpoints_malik_T_150_bs_100_tg_100_initial_lr_0.001_clipnorm_25.0_nschd_[250,500.0,1000.0,1300.0,2000.0,2500.0,3300.0]_nrate_[0.01,0.05,0.1,0.2,0.3,0.5,0.68]_decschd_[1500.0,4500.0]_decrate_[0.1,0.1]_fs_eating';
filenames{8,1} = 'checkpoints_malik_T_150_bs_100_tg_100_initial_lr_0.001_clipnorm_25.0_nschd_[250,500.0,1000.0,1300.0,2000.0,2500.0,3300.0]_nrate_[0.01,0.05,0.1,0.2,0.3,0.5,0.7]_decschd_[1500.0,4500.0]_decrate_[0.1,0.1]_fs_discussion';

filenames{9,1} = 'checkpoints_lstm_T_150_bs_100_tg_100_initial_lr_0.001_clipnorm_25.0_nschd_[250,500.0,1000.0,1300.0,2000.0,2500.0,3300.0]_nrate_[0.01,0.05,0.1,0.2,0.3,0.5,0.7]_decschd_[1500.0,4500.0]_decrate_[0.1,0.1]_fs_final';
filenames{10,1} = 'checkpoints_lstm_T_150_bs_100_tg_100_initial_lr_0.001_clipnorm_25.0_nschd_[250,500.0,1000.0,1300.0,2000.0,2500.0,3300.0]_nrate_[0.01,0.05,0.1,0.2,0.3,0.5,0.7]_decschd_[1500.0,4500.0]_decrate_[0.1,0.1]_fs_smoking';
filenames{11,1} = 'checkpoints_lstm_T_150_bs_100_tg_100_initial_lr_0.001_clipnorm_25.0_nschd_[250,500.0,1000.0,1300.0,2000.0,2500.0,3300.0]_nrate_[0.01,0.05,0.1,0.2,0.3,0.5,0.7]_decschd_[1500.0,4500.0]_decrate_[0.1,0.1]_fs_eating';
filenames{12,1} = 'checkpoints_lstm_T_150_bs_100_tg_100_initial_lr_0.001_clipnorm_25.0_nschd_[250,500.0,1000.0,1300.0,2000.0,2500.0,3300.0]_nrate_[0.01,0.05,0.1,0.2,0.3,0.5,0.7]_decschd_[1500.0,4500.0]_decrate_[0.1,0.1]_fs_discussion';




filenames{1,2} = 16;
filenames{2,2} = 0;
filenames{3,2} = 16;
filenames{4,2} = 16;

filenames{1,3} = []; %[0,1,3,4,6,7]; %walking
filenames{1,4} = []; %[15]; %eating
filenames{1,5} = [18,22]; %smoking

filenames{2,5} = [0,1,4,6,7]; % smoking
filenames{3,4} = []; %[0,1,2,3,5,6,7]; % eating
filenames{4,6} = []; %[1,2,3]; % discussion

filenames{1,7} = 4000;
filenames{2,7} = 4500;
filenames{3,7} = 4000;
filenames{4,7} = 4250;

filenames{5,7} = 5000;
filenames{6,7} = 5000;
filenames{7,7} = 5000;
filenames{8,7} = 5000;

filenames{9,7} = 5000;
filenames{10,7} = 5000;
filenames{11,7} = 5000;
filenames{12,7} = 5000;

destination_activity{3} = 'walking';
destination_activity{4} = 'eating';
destination_activity{5} = 'smoking';
destination_activity{6} = 'discussion';

for i = 1:size(filenames,1)
    
    idx = mod(i-1,4) + 1;
    
    iteration = filenames{i,7};
    for k = 3:6
        % {walking,eating,smoking,discussion}
        activity_id = filenames{idx,k};
        for j = 1:size(activity_id,2)
            playGeneratedMotion( filenames{i,1},['forecast_iteration_',num2str(iteration),'_N_',num2str(activity_id(j)),'.dat'],5,false,filenames{idx,2});
            if i <= 4
                copy_to = ['/media/ashesh/ssd2/DRA/user_study/dra/',destination_activity{k},'/.'];
                unix(['mkdir -p ',copy_to]);
                copy_to = ['/media/ashesh/ssd2/DRA/user_study/dra/',destination_activity{k},'/',num2str(activity_id(j)),'_crop.avi'];
                copy_from = [filenames{i,1},'/videos/forecast_iteration_',num2str(iteration),'_N_',num2str(activity_id(j)),'.avi'];
                unix(['mv ',copy_from,' ',copy_to]);  
                
                playGeneratedMotion( filenames{i,1},['ground_truth_forecast_N_',num2str(activity_id(j)),'.dat'],5,false,filenames{idx,2});
                copy_to = ['/media/ashesh/ssd2/DRA/user_study/gt/',destination_activity{k},'/.'];
                unix(['mkdir -p ',copy_to]);
                copy_to = ['/media/ashesh/ssd2/DRA/user_study/gt/',destination_activity{k},'/',num2str(activity_id(j)),'_crop.avi'];
                copy_from = [filenames{i,1},'/videos/ground_truth_forecast_N_',num2str(activity_id(j)),'.avi'];
                unix(['mv ',copy_from,' ',copy_to]);  
            elseif i <= 8
                copy_to = ['/media/ashesh/ssd2/DRA/user_study/erd/',destination_activity{k},'/.'];
                unix(['mkdir -p ',copy_to]);
                copy_to = ['/media/ashesh/ssd2/DRA/user_study/erd/',destination_activity{k},'/',num2str(activity_id(j)),'_crop.avi'];
                copy_from = [filenames{i,1},'/videos/forecast_iteration_',num2str(iteration),'_N_',num2str(activity_id(j)),'.avi'];
                unix(['mv ',copy_from,' ',copy_to]);  
            else
                copy_to = ['/media/ashesh/ssd2/DRA/user_study/lstm/',destination_activity{k},'/.'];
                unix(['mkdir -p ',copy_to]);
                copy_to = ['/media/ashesh/ssd2/DRA/user_study/lstm/',destination_activity{k},'/',num2str(activity_id(j)),'_crop.avi'];
                copy_from = [filenames{i,1},'/videos/forecast_iteration_',num2str(iteration),'_N_',num2str(activity_id(j)),'.avi'];
                unix(['mv ',copy_from,' ',copy_to]);                  
            end;
        end;
    end;
end;