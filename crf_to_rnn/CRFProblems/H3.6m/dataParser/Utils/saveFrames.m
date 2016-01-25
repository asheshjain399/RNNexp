function  saveFrames( N )

    mocapDir = {};
    mocapDir{1} = 'checkpoints_lstm_T_150_bs_100_tg_100_initial_lr_0.001_clipnorm_25.0_nschd_[250,500.0,1000.0,1300.0,2000.0,2500.0,3300.0]_nrate_[0.01,0.05,0.1,0.2,0.3,0.5,0.7]_decschd_[1500.0,4500.0]_decrate_[0.1,0.1]_fs_final';
    mocapDir{2} = 'checkpoints_malik_T_150_bs_100_tg_100_initial_lr_0.001_clipnorm_25.0_nschd_[250,500.0,1000.0,1300.0,2000.0,2500.0,4000.0]_nrate_[0.01,0.05,0.1,0.2,0.3,0.5,0.65]_decschd_[1500.0,4500.0]_decrate_[0.1,0.1]_fs_final';
    mocapDir{3} = 'checkpoints_dra_T_150_bs_100_tg_100_ls_512_fc_256_initial_lr_0.001_clipnorm_25.0_nschd_[250,500.0,1000.0,1300.0,2000.0,2500.0,3300.0]_nrate_[0.01,0.05,0.1,0.2,0.3,0.5,0.7]_decschd_[1500.0,4500.0]_decrate_[0.1,0.1]_fs_final';

    checkpoint = {};
    checkpoint{1} = 5000;
    checkpoint{2} = 5000;
    checkpoint{3} = 4000;

    playGeneratedMotion( mocapDir{3},['ground_truth_forecast_N_',num2str(N),'.dat'],5,true );
    
    for i = 1:size(checkpoint,2)
        playGeneratedMotion( mocapDir{i},['forecast_iteration_',num2str(checkpoint{i}),'_N_',num2str(N),'.dat'],5,false );              
    end;
end

