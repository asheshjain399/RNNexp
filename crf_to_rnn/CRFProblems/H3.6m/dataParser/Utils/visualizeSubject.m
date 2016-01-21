function visualizeSubject( subject,action,subaction )
    addpaths;
    db = H36MDataBase.instance;
    angleSkel = getAnglesSkel(db,subject);
    poseData = H36MPoseDataAcess(['/home/ashesh/Downloads/H3.6/S' num2str(subject) '/MyPoseFeatures/D3_Angles/' getFileName(db,subject,action,subaction) '.cdf']);
    [expmap_skel,expmap_channels] = getExpmapFromSkeleton(angleSkel,poseData.Block);
    [channels_in_local_coordinates,expmap_skel,R0,T0] = changeCoordinateSpace(angleSkel,poseData);
    playGeneratedMotion( '',subject,channels_in_local_coordinates )
    rootRotInd = expmap_skel.tree(1).expmapInd;
    data = expmap_channels(1,:);
    R0 = expmap2rotmat(data(rootRotInd));
    disp('Initial rotation w.r.t. world coordinate axis');
    disp(R0);
    expPlayData(expmap_skel, expmap_channels(1,:), 1.0/120);
end

