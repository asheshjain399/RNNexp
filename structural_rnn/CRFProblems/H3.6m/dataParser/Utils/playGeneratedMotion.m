function playGeneratedMotion( model,filename,subject,save_frames,thresh )
    addpaths;
    str_split = strsplit(filename,'_');
    motionidx = str_split{end};
    db = H36MDataBase.instance;
    
    temp = strsplit(motionidx,'.');    
    Nval = str2num(temp{1});
    actionidx = Nval + 1;
    
    if nargin < 4
        save_frames = false;
    end;
    
    if nargin < 5
        thresh = 16;
    end;
    
    
    tt = 'forecastidx1.dat';
    if Nval >= thresh
        tt = 'forecastidx.dat';
    end;
    
    if exist([model,'/',tt],'file') == 2
        forecast = dlmread([model,'/forecastidx.dat'],',');
        
        if actionidx > size(forecast,1)
            return;
        end;
        
        [ R0,T0 ] = getGroundTruthR0T0( db,subject,forecast(actionidx,2),forecast(actionidx,4),forecast(actionidx,3),'even' );
        [ R_hat,T_hat ] = getGroundTruthR0T0( db,subject,forecast(actionidx,2),forecast(actionidx,4),1,'even' );
        %R_hat = [1 0 0; 0 0 -1; 0 1 0];
        R_hat = R_hat';
        R0 = R0*R_hat;
        T0 = (R_hat * T0')';
    else
        R0 = eye(3);
        T0 = [0 0 0];        
    end;

    idx = 'long';
    
    
    angleSkel = getAnglesSkel(db,subject);
    c = zeros(1,78);
    [ skel_expmap,~ ] = getExpmapFromSkeleton( angleSkel,c );    
    motionidx_file = [model,'/','motionprefix',idx,'_N_',motionidx];
    forecast_file = [model,'/',filename];
    
    frame_dir = [model,'/videos/',filename,'_'];
    
    f = [];
    if exist(motionidx_file,'file') == 2
        f1=csvread(motionidx_file);
        f = f1;
    end;
    if exist(forecast_file,'file') == 2
        f2=csvread(forecast_file);
        f = [f ; f2];
    end;
    %if exist(forecast_file,'file') == 2 && exist(motionidx_file,'file') == 2
    if size(f,1) > 0
        %f1=csvread(motionidx_file);
        %f2=csvread(forecast_file);
        %f = [f1;f2];

        video_dir = [model,'/videos'];
        unix(['mkdir -p ',video_dir]);
        video_filename = [video_dir,'/',filename(1:end-4),'.avi'];
        %if exist(video_filename,'file') ~= 2
            writerObj = VideoWriter(video_filename);
            writerObj.Quality = 100;
            %writerObj = VideoWriter(video_filename);
            writerObj.FrameRate = 20.0;
            open(writerObj);
            %[R0,T0,writerObj] = playVideo(motionidx_file,subject,R0,T0,writerObj,skel_expmap,false);
            playVideo(f,R0,T0,writerObj,skel_expmap,frame_dir,save_frames);
            close all;
        %end;
    end;
end

function [R0,T0,writerObj] = playVideo(f,R0,T0,writerObj,skel_expmap,frame_dir,save_frames)
    %f=csvread(filename);
    [channels_reconstruct,R0,T0] = revertCoordinateSpace(f,R0,T0);
    if save_frames
        expPlayData2(skel_expmap, channels_reconstruct, 1.0/100,writerObj,frame_dir);
    else
        expPlayData2(skel_expmap, channels_reconstruct, 1.0/100,writerObj);
    end;
end