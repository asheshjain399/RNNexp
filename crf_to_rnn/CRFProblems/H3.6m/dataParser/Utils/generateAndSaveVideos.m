function  generateAndSaveVideos(  )
    addpaths;
    close all;
    names = dir;
    db = H36MDataBase.instance;
    for i = 1:numel(names)
        nm = names(i);
        if nm.isdir && ~strcmp(nm.name , '.') && ~strcmp(nm.name , '..')
            disp(nm.name);
            if ~strcmp(nm.name,'checkpoints_malik_T_150_bs_100_tg_100_initial_lr_0.001_clipnorm_25.0_nschd_[900.0,2000.0,3000.0,4000.0,5000.0,6000.0,7000.0,8000.0]_nrate_[0.1,0.3,0.5,0.7,0.8,0.9,1.0,1.2]_decschd_[1500.0,4500.0]_decrate_[0.1,0.1]_fs')
                continue;
            end;
            %unix(['mkdir -p ',nm.name,'/videos']);
            for N = 0:1:25
                
                playGeneratedMotion( nm.name,['ground_truth_forecast_N_',num2str(N),'.dat'],11 );
                %{
                forecast_file = [nm.name, '/ground_truth_forecast_N_',num2str(N),'.dat'];
                if exist(forecast_file,'file') == 2
                    video_filename = [nm.name, '/videos/ground_truth_forecast_N_',num2str(N),'.avi'];
                    if exist(video_filename,'file') ~= 2
                        disp('ground_truth');
                        saveVideo(forecast_file,video_filename,db);            
                        
                    end;
                end;
                
                forecast_file = [nm.name, '/motionprefix_N_',num2str(N),'.dat'];
                if exist(forecast_file,'file') == 2
                    video_filename = [nm.name, '/videos/motionprefix_N_',num2str(N),'.avi'];
                    if exist(video_filename,'file') ~= 2
                        saveVideo(forecast_file,video_filename,db);                            
                    end;
                end;
                %}
                for iterations = 4750
                    playGeneratedMotion( nm.name,['forecast_iteration_',num2str(iterations),'_N_',num2str(N),'.dat'],11 );              
                    %{
                    forecast_file = [nm.name, '/forecast_iteration_',num2str(iterations),'_N_',num2str(N),'.dat'];  
                    if exist(forecast_file,'file') == 2
                        video_filename = [nm.name, '/videos/forecast_epoch_',num2str(epoch),'_N_',num2str(N),'.avi'];
                        if exist(video_filename,'file') ~= 2
                            disp(['generation epoch=',num2str(epoch)]);
                            saveVideo(forecast_file,video_filename,db);
                        end;
                    end;
                    %}
                end;
            end;
        end;
        
    end;

end

function saveVideo(forecast_file,video_filename,db)
    f=csvread(forecast_file);
    R0 = eye(3);
    T0 = [0 0 0];
    f(:,1:6)=0;
    channels_reconstruct = revertCoordinateSpace(f,R0,T0);
    angleSkel = getAnglesSkel(db,11);
    c = zeros(1,78);
    [ skel_expmap,~ ] = getExpmapFromSkeleton( angleSkel,c );
    writerObj = VideoWriter(video_filename);
    writerObj.FrameRate = 100.0;
    open(writerObj);
    try
        expPlayData(skel_expmap, channels_reconstruct, 1.0/100, writerObj);  
    catch
        disp(['not saved ',video_filename]);
    end;
    close all;
    pause(1.0);
end




