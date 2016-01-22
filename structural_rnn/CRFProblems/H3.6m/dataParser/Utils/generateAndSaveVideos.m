function  generateAndSaveVideos(  )
    addpaths;
    close all;
    names = dir;
    db = H36MDataBase.instance;
    for i = 1:numel(names)
        nm = names(i);
        if nm.isdir && ~strcmp(nm.name , '.') && ~strcmp(nm.name , '..')
            disp(nm.name);
            
            if ~strcmp(nm.name,'checkpoints_dra_T_150_bs_100_tg_100_ls_512_fc_256_initial_lr_0.001_clipnorm_25.0_nschd_[250,500.0,1000.0,1300.0,2000.0,2500.0,3300.0]_nrate_[0.01,0.05,0.1,0.2,0.3,0.5,0.7]_decschd_[1500.0,4500.0]_decrate_[0.1,0.1]_fc_fs_final')
                continue;
            end;
            
            %if ~strcmp(nm.name,'checkpoints_lstm_T_150_bs_100_tg_100_initial_lr_0.001_clipnorm_25.0_nschd_[250,500.0,1000.0,1300.0,2000.0,2500.0,3300.0]_nrate_[0.01,0.05,0.1,0.2,0.3,0.5,0.7]_decschd_[1500.0,4500.0]_decrate_[0.1,0.1]_fs_final')
            %    continue;
            %end;
            
            %if ~strcmp(nm.name,'checkpoints_lstm_T_150_bs_100_tg_100_initial_lr_0.001_clipnorm_25.0_nschd_[250,500.0,1000.0,1300.0,2000.0,2500.0,3300.0]_nrate_[0.01,0.05,0.1,0.2,0.3,0.5,0.7]_decschd_[1500.0,4500.0]_decrate_[0.1,0.1]_fs_smoking')
            %    continue;
            %end;
            
            %unix(['mkdir -p ',nm.name,'/videos']);
            for N = 0:23 %23
                
                playGeneratedMotion( nm.name,['ground_truth_forecast_N_',num2str(N),'.dat'],5,false );
                %for iterations = 5000
                %        playGeneratedMotion( nm.name,['forecast_iteration_',num2str(iterations),'_N_',num2str(N),'.dat'],5,false );              
                %end;
                %figure; plotCells( N );
            end;
        end;
        
    end;

end





