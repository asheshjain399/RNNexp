%{
Converts pose data from Euler angle to euclidean map space and center it
w.r.t to starting position
%}


function [channels_in_local_coordinates,skel,R0,T0] = changeCoordinateSpace(angleSkel,poseData)
    addpaths;
    [skel,channels] = getExpmapFromSkeleton(angleSkel,poseData.Block);
    channels_in_local_coordinates = channels;
    rootRotInd = skel.tree(1).expmapInd;
    data = channels(1,:);
    R_prev = expmap2rotmat(data(rootRotInd));
    T_prev = data(1:3);
    R0 = R_prev;
    T0 = T_prev;

    for ii=1:size(channels,1)
        data = channels(ii,:);
        R = expmap2rotmat(data(rootRotInd));
        T = data(1:3);

        R_diff = R*(R_prev^-1);
        T_diff = T - T_prev;
        T_diff_projected = R_prev*T_diff';
        %T_diff_projected = R_diff*T_diff';
        expmap_diff = rotmat2expmap(R_diff);

        channels_in_local_coordinates(ii,1:3) = T_diff_projected;
        channels_in_local_coordinates(ii,rootRotInd) = expmap_diff;

        R_prev = R;
        T_prev = T;
    end;

end


%{
function [channels_in_local_coordinates,skel,R0,T0] = changeCoordinateSpace(angleSkel,poseData)
    addpaths;
    [skel,channels] = getExpmapFromSkeleton(angleSkel,poseData.Block);
    channels_in_local_coordinates = channels;
    rootRotInd = skel.tree(1).expmapInd;
    data = channels(1,:);
    R_prev = expmap2rotmat(data(rootRotInd));
    T_prev = data(1:3);
    R0 = R_prev;
    T0 = T_prev;

    channels_in_local_coordinates(2:end,4) = poseData.Block(2:end,4) - poseData.Block(1:end-1,4); % incremental z-rotation
    channels_in_local_coordinates(1,4) = 0.0;
    
    for ii=1:size(channels,1)
        data = channels(ii,:);
        R = expmap2rotmat(data(rootRotInd));
        T = data(1:3);

        R_diff = R*(R_prev^-1);
        T_diff = T - T_prev;
        T_diff_projected = R_diff*T_diff';
        expmap_diff = rotmat2expmap(R_diff);
        
        eu = RotMat2Euler(R_diff);
        channels_in_local_coordinates(ii,1) = T_diff_projected(3); %incremental forward movement
        channels_in_local_coordinates(ii,2) = T_diff_projected(1); %incremental sideward movement
        channels_in_local_coordinates(ii,5) = eu(1); %yaw
        channels_in_local_coordinates(ii,6) = eu(3); %roll
        
        R_prev = R;
        T_prev = T;
    end;

end


function [channels_reconstruct,R_prev,T_prev] = revertCoordinateSpace(channels_self,R0,T0)
    addpaths;
    channels_reconstruct = channels_self;
    R_prev = R0;
    T_prev = T0;
    rootRotInd = 4:6;
    for ii = 1:size(channels_self,1)
        eu(1) = channels_self(ii,5);
        eu(2) = channels_self(ii,4);
        eu(3) = channels_self(ii,6);
        R_diff = Euler2RotMat(eu);
        T_diff_projected(3) = channels_self(ii,1);
        T_diff_projected(1) = channels_self(ii,2);
        
        if ii == 1
            T_diff_projected(2) = 0.0;
        else
            T_diff_projected(2) = channels_self(ii,3) - channels_self(ii-1,3);
        end;
        
        R = expmap2rotmat(channels_self(ii,rootRotInd))*R_prev;
        channels_reconstruct(ii,rootRotInd) = rotmat2expmap(R);
        T = T_prev + ((R_prev^-1)*(channels_self(ii,1:3))')';
        channels_reconstruct(ii,1:3) = T;
        T_prev = T;
        R_prev = R;
    end;
end

%}