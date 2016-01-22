% Version 1.01 
%
% Code provided by Graham Taylor, Geoff Hinton and Sam Roweis 
%
% For more information, see:
%     http://www.cs.toronto.edu/~gwtaylor/publications/nips2006mhmublv
%
% Permission is granted for anyone to copy, use, modify, or distribute this
% program and accompanying programs and documents for any purpose, provided
% this copyright notice is retained and prominently displayed, along with
% a note saying that the original programs are available from our
% web page.
% The programs and documents are distributed without any warranty, express or
% implied.  As the programs were written for research purposes only, they have
% not been tested to the degree that would be advisable in any important
% application.  All use of these programs is entirely at the user's own risk.
%
% This program preprocesses data (stage 1 of 2)
% We need to unwrap the rotation about the vertical axis
% before we take the difference in the next stage of preprocessing
%
% We support two types of skeletons:
%  1) Those built from the CMU database (acclaim)
%     http://mocap.cs.cmu.edu/
%  2) Those built from data from Eugene Hsu (mit)
%     http://people.csail.mit.edu/ehsu/work/sig05stf/
%
% The program assumes that the following variables are set externally:
% n1     -- order of the first layer CRBM
% Motion -- cell array of motion data
% skel   -- skeleton structure

newrep = []; %the "body-centred" representation for orientation
grdvel = []; %the relative-to-body representation for x,y velocity
             %the third component is the delta for newrep(3), i.e. about z               
 myvec = [0 0 1]; %straight ahead
 myvec2 = [1 0 0]; %out right hip
 vertvec = [0 1 0]; %positive vertical axis
 %find the rotation matrix based on the root Exponential map angles
 %for this frame
 rootRotInd = skel.tree(1).expmapInd;
             


  %BEGIN FIRST PART - BODY CENTRED COORDINATES  
  for ii=1:size(channels,1)
    
    data = channels(ii,:); %single frame    
  
      R = expmap2rotmat(data(rootRotInd));

      %what do our normal vectors become?
    newvec = myvec*R; %calculate myvec rotated by the root orientation
    newvec2 = myvec2*R;
    
    %record the angles the new normal vectors make with the -z axis    
    ct = (newvec*-vertvec')/norm(newvec);    
    %Motion{jj}(ii,1) = acos(ct);  
    newrep{jj}(ii,1) = acos(ct);
    ct2 = (newvec2*-vertvec')/norm(newvec2);
    %Motion{jj}(ii,2) = acos(ct2);
    newrep{jj}(ii,2) = acos(ct2);
        
    %We want our "about" vertical to be meaningful, so the angle being
    %stored is the "change" from the "rest" position
    %Since atan2 always works with the x axis as its base angle, for the
    %CMU data (with the subject by default facing the z-axis), 
    %we will measure newvec2
    %With the MIT data, "rest" position is to face straight ahead in the
    %x-axis, so we naturally measure newvec
    %projection is simply the x and z components
    if strcmp(skel.type,'acclaim') || strcmp(skel.type,'bvh') || ...
        strcmp(skel.type,'cmubvh')
      %CMU DATA
      
      %Note that angle will be calculated from the x axis
      %In the CMU case, this is newvec2 (in the zero position)
      ux = newvec2(1);
      uz = newvec2(3);
    elseif strcmp(skel.type,'mit')
      %MIT DATA
      ux = newvec(1);
      uz = newvec(2);
    else
      error('Unknown skeleton type');
    end
    
    %Using the atan2 function we can calculate the angle between -pi and pi
    %given the horizontal and vertical components        

    newrep{jj}(ii,3) = atan2(uz,ux);
  end
  
  %END FIRST PART - BODY CENTRED COORDINATES
  %BEGIN SECOND PART - GROUND PLANE VECTORS
  
  %Since the atan2 function returns an angle between -pi and pi, we now do
  %phase unwrapping to recognize the fact that within a sequence we may
  %turn more than 2pi radians
  %The unwrapping maintains the continuity of the turn  
  newrep{jj}(:,3) = unwrap(newrep{jj}(:,3));

  for ii=1:size(channels,1)

    if ii<size(channels,1)
      %store differences - dimension 3: about-vertical rotation
      
      if strcmp(skel.type,'acclaim') || strcmp(skel.type,'bvh') || ...
          strcmp(skel.type,'cmubvh')
        %CMU DATA
        % dimension 1: - x position, dimension 3: z position
        mydif = [newrep{jj}(ii+1,3) channels(ii+1,[1 3])] - ...
          [newrep{jj}(ii,3) channels(ii,[1 3])];
      elseif strcmp(skel.type,'mit')
        %MIT DATA
        % dimension 4: - x position, dimension 5: y position
        mydif = [newrep{jj}(ii+1,3) channels(ii+1,4:5)] - ...
          [newrep{jj}(ii,3) channels(ii,4:5)];
      else
        error('Unknown skeleton type');
      end      
            
      %We also want to represent our horizontal position in terms of the
      %body-centred coordinate system
      mh = norm([mydif(2) mydif(3)]); %mag  of the horizontal movemt vector
      %angle of the horizontal position vector (counterclk from x axis)
      apos = atan2(mydif(3),mydif(2));
      %angle of the normal vector straight out of the body (projection in xz)
      %No need to restrict it to 0,2*pi (or any range)
      %Nor do we need to restrict the offset to any range
      %As the offset (whatever it is, goes through cos & sin)
      anorm = newrep{jj}(ii,3);
      
      %For MIT, the u (cos) component is the component in line with the
      %"straight out of body" vector
      %For CMU, the u (cos) component is the compoent in line with the
      %"straight out of right hip" vector      
      %u component of movement vector 
      grdvel{jj}(ii,1) = mh*cos(anorm - apos);
      %v component of movement vector 
      grdvel{jj}(ii,2) = mh*sin(anorm - apos);
      
      %Store the differences in the data cell for the about-vertical rot
      grdvel{jj}(ii,3) = mydif(1);
    else
      %don't care about the last frame, because we're taking differences
      grdvel{jj}(ii,1:3) = grdvel{jj}(ii-1,1:3);
    end
  end

  %Note that the last frame has values identical to frame (end-1)
  %This isn't the nicest thing to do
  
  %Overwrite the training data (Motion) with the new representations
  %Even if it is CMU data, let's overwrite the first 6 dimensions  
  
  if any(strcmp(skel.type,{'acclaim','bvh','cmubvh'}))
   %CMU DATA
   %Absolute vertical position - dimension 2 is the one we don't want to
   %touch
   channels(:,6) = channels(:,2);   
  elseif strcmp(skel.type,'mit')
    %MIT DATA
    %The absolute vertical is already in dimension 6
  else
    error('Unknown skeleton type');
  end

  channels(:,1:2) = newrep{jj}(:,1:2); %body-centred orientations
  channels(:,3) = grdvel{jj}(:,3);     %delta about-z orientation
  channels(:,4:5) = grdvel{jj}(:,1:2); %ground plane relative velocities      
