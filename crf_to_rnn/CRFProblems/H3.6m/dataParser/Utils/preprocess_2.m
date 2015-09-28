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
            [channels_in_local_coordinates,skel,R0,T0] = changeCoordinateSpace(angleSkel,poseData);
            channels_coordinate = [channels_coordinate;channels_in_local_coordinates];
            dlmwrite(['/home/ashesh/Downloads/H3.6/S' num2str(subject) '/MyPoseFeatures/D3_Angles/' actionsToName{action} '_' num2str(subaction) '.txt'],channels_in_local_coordinates,'delimiter',',','precision','%4.7f');
        end;
    end;
end;
            
end


%skelPlayData(angleSkel_1, poseData.Block, 1.0/120);
%expPlayData(angleSkel_expmap, expmapchannels, 1.0/120);
