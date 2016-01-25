function [ R0,T0 ] = getGroundTruthR0T0( db,subject,action,subaction,frame,parity )
    addpaths;
    poseData = H36MPoseDataAcess(['/home/ashesh/Downloads/H3.6/S' num2str(subject) '/MyPoseFeatures/D3_Angles/' getFileName(db,subject,action,subaction) '.cdf']);
    angleSkel = getAnglesSkel(db,subject);
    if strcmp(parity,'even')
        idx = 2*frame + 1;
    elseif strcmp(parity,'odd')
        idx = 2*(frame + 1);
    end;
    poseData.Block = poseData.Block(idx,:);
    [~,~,R0,T0] = changeCoordinateSpace(angleSkel,poseData);
end

