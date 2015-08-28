%clear all; clc;
function preprocess_2()
addpaths;

db = H36MDataBase.instance;

subjects = [5,6,7,8,9,11,1];
actions = 2:16;
subactions = 1:2;
cameras = 1:4;

actionsToName={'','directions','discussion','eating','greeting','phoning','posing','purchases','sitting','sittingdown','smoking','takingphoto','waiting','walking','walkingdog','walkingtogether'};

channels_coordinate = [];
for subject = subjects
    angleSkel = getAnglesSkel(db,subject);
    for action = actions
        for subaction = subactions
            %disp(getFileName(db,subject,action,subaction))
            poseData = H36MPoseDataAcess(['/home/ashesh/Downloads/H3.6/S' num2str(subject) '/MyPoseFeatures/D3_Angles/' getFileName(db,subject,action,subaction) '.cdf']);
            [channels_in_local_coordinates,skel,R0,T0] = changeCoordinateSpace(db,angleSkel,poseData);
            channels_coordinate = [channels_coordinate;channels_in_local_coordinates];
            dlmwrite(['/home/ashesh/Downloads/H3.6/S' num2str(subject) '/MyPoseFeatures/D3_Angles/' actionsToName{action} '_' num2str(subaction) '.txt'],channels_in_local_coordinates,'delimiter',',','precision','%4.7f');
        end;
    end;
end;
            
end

function [channels_in_local_coordinates,skel,R0,T0] = changeCoordinateSpace(db,angleSkel,poseData)
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
        expmap_diff = rotmat2expmap(R_diff);

        channels_in_local_coordinates(ii,1:3) = T_diff_projected;
        channels_in_local_coordinates(ii,rootRotInd) = expmap_diff;

        R_prev = R;
        T_prev = T;
    end;

end

function channels_reconstruct = revertCoordinateSpace(channels_in_local_coordinates,R0,T0)
    channels_reconstruct = channels_in_local_coordinates;
    R_prev = R0;
    T_prev = T0;
    rootRotInd = 4:6;
    for ii = 1:size(channels_self,1)
        R = expmap2rotmat(channels_self(ii,rootRotInd))*R_prev;
        channels_reconstruct(ii,rootRotInd) = rotmat2expmap(R);
        T = T_prev + ((R_prev^-1)*(channels_self(ii,1:3))')';
        channels_reconstruct(ii,1:3) = T;
        T_prev = T;
        R_prev = R;
    end;
end
%skelPlayData(angleSkel_1, poseData.Block, 1.0/120);
%expPlayData(angleSkel_expmap, expmapchannels, 1.0/120);
