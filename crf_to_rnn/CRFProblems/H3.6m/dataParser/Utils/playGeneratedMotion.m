function playGeneratedMotion( model,filename,subject,f,R0,T0 )
    addpaths;
    if nargin < 4   
        R0 = eye(3);
        T0 = [0 0 0];
    end;
    
    db = H36MDataBase.instance;
    angleSkel = getAnglesSkel(db,subject);
    c = zeros(1,78);
    [ skel_expmap,~ ] = getExpmapFromSkeleton( angleSkel,c );    
    str_split = strsplit(filename,'_');
    motionidx = str_split{end};
    motionidx_file = [model,'/','motionprefix_N_',motionidx];
    forecast_file = [model,'/',filename];
    
    if exist(forecast_file,'file') == 2 && exist(motionidx_file,'file') == 2
        f1=csvread(motionidx_file);
        f2=csvread(forecast_file);
        f = [f1;f2];

        video_dir = [model,'/videos'];
        unix(['mkdir -p ',video_dir]);
        video_filename = [video_dir,'/',filename(1:end-4),'.avi'];
        if exist(video_filename,'file') ~= 2
            writerObj = VideoWriter(video_filename);
            writerObj.FrameRate = 20.0;
            open(writerObj);
            %[R0,T0,writerObj] = playVideo(motionidx_file,subject,R0,T0,writerObj,skel_expmap,false);
            playVideo(f,R0,T0,writerObj,skel_expmap,true);
            close all;
        end;
    end;
end

function [R0,T0,writerObj] = playVideo(f,R0,T0,writerObj,skel_expmap,close_before_returning)
    %f=csvread(filename);
    [channels_reconstruct,R0,T0] = revertCoordinateSpace(f,R0,T0);
    expPlayData(skel_expmap, channels_reconstruct, 1.0/100,writerObj,close_before_returning);
end