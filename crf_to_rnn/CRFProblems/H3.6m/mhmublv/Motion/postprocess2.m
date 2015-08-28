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
% This program postprocesses data. 
% It is the second stage of preprocessing (essentially reverses
% preprocess1.m)
%
% We support two types of skeletons:
%  1) Those built from the CMU database (acclaim)
%     http://mocap.cs.cmu.edu/
%  2) Those built from data from Eugene Hsu (mit)
%     http://people.csail.mit.edu/ehsu/work/sig05stf/
%

%%%%%%%%%%
% Note our convention (because I started this way with the CMU code)
% x- ground plane axis (looking at screen, going to the right)
% z- ground plane axis (looking at screen, going "in")
% y- vertical axis
% For MIT data, subject at 0 position is facing positive x axis
% channel 1 - x component (exponential map)
% channel 2 - z component (exponential map)
% channel 3 - y (vertical) component (exponential map)
% Note this is the way that MATLAB expects it -- for the CMU data, we need
%   to reverse channels 2,3 when we plot because the ordering is (x,y,z)
%%%%%%%%%%%%%%%%

%We unwrapped before, so that deltas could be calculated for the rotation
%about the vertical axis
%But now we want to wrap back to -pi < theta =< pi so before we call
%expmap2rotmat

%Extract relative angles
phi = newdata(:,1);         % rotation along anteroposterior
theta = newdata(:,2);       % rotation along mediolateral

%Extract the deltas
vertrotdelta = newdata(:,3);    % angular velocity along the vertical axis
groundxdelta = newdata(:,4);    % anterior component of the velocity
groundydelta = newdata(:,5);    % lateral component of the velocity

%Remember original dimensions are different for CMU and MIT
if strcmp(skel.type,'acclaim') || strcmp(skel.type,'bvh') || ...
    strcmp(skel.type,'cmubvh')
  %CMU DATA
  pos_x = skel.tree(1).posInd(1);
  pos_y = skel.tree(1).posInd(2);
  pos_z = skel.tree(1).posInd(3);
  rot_x = skel.tree(1).expmapInd(1);
  rot_y = skel.tree(1).expmapInd(2);
  rot_z = skel.tree(1).expmapInd(3);
  
  %For CMU, the absolute position is saved in a different dimension than it
  %originally was in; We need to write it back to this dimension
  newdata(:,pos_y) = newdata(:,6);  % vertical position
  
elseif strcmp(skel.type,'mit')
  %MIT DATA
  pos_x = skel.tree(1).offset(1);
  pos_z = skel.tree(1).offset(2);
  pos_y = skel.tree(1).offset(3);
  rot_x = skel.tree(1).or(1);
  rot_z = skel.tree(1).or(2);
  rot_y = skel.tree(1).or(3);
else
  error('Unknown skeleton type');
end

%%% FIRST FRAME
%Assuming we start all at zero (could put in initial conditions here)

%if we have defined the facing variable
%set this in radians
if exist('facing','var')
  newdata(1,rot_y) = facing;% could be, for example,  newrep{1}(1,3)
else
  newdata(1,rot_y) = 0;
end
newdata(1,pos_x) = 0;
newdata(1,pos_z) = 0;
%%% END FIRST FRAME

%STEP ONE - CONVERT FROM DELTAS TO REAL VALUES
%           PROCESS THE GROUND PLANE TRANSLATIONS

%loop through each frame
for ii=2:size(newdata,1)
  %We have horizontal movements (deltas) in the body-centred reference
  %frame
  %need to convert to horizontal positions in standard reference frame
  %orientation first (cumulative sum)
  newdata(ii,rot_y) = newdata(ii-1,rot_y) + vertrotdelta(ii-1,1);
  
  %now position
  %first, the magnitude
  m = norm([groundxdelta(ii-1,1) groundydelta(ii-1,1)]);
  %calculate the offset
  dab = atan2(groundydelta(ii-1,1),groundxdelta(ii-1,1));
  be = newdata(ii-1,rot_y); %taken with respect to the starting frame
  %al is the original orientation of the velocity vector (delta)
  al = be - dab;
  newdata(ii,pos_x) = newdata(ii-1,pos_x) + m*cos(al); %add x component
  newdata(ii,pos_z) = newdata(ii-1,pos_z) + m*sin(al); %add z component
end

%STEP TWO - CONVERT FROM BODY-CENTRED ORIENTATIONS TO EXPMAPS
%loop through each frame
for ii=1:size(newdata,1)
  %for rotation about the vertical, work in radians
  %wrapping to 0,2pi makes it easier
  psi = mod(newdata(ii,rot_y),2*pi);

  %Remember what was being stored as "angle about the vertical"
  %CMU - the right hip's offset from the x-axis
  %MIT - the "straight ahead's" offset from the x-axis
  %Here, we will let u refer to this vector (either hip or straight ahead)
  %v will be the other vector
  %w will point straight up
  %this will describe the rotation, and from these axes we can calculate
  %the rotation matrix and then Euler angles

  if strcmp(skel.type,'acclaim') || strcmp(skel.type,'bvh') || ...
    strcmp(skel.type,'cmubvh')
    %CMU DATA
    uy = -cos(theta(ii));
  elseif strcmp(skel.type,'mit')
    uy = -cos(phi(ii)); %easy
  else
    error('Unknown skeleton type');
  end  
  
  %finding x and z components of u trickier, as it depends on the quadrant
  if psi < pi/2
    uz_over_ux = tan(psi);
    magux = sqrt((1 - uy^2)/(1 + uz_over_ux^2));
    maguz = magux*uz_over_ux;
    uz = maguz; %pos
    ux = magux; %pos
  elseif psi < pi
    uz_over_ux = tan(pi - psi);
    magux = sqrt((1 - uy^2)/(1 + uz_over_ux^2));
    maguz = magux*uz_over_ux;
    uz = maguz;  %pos
    ux = -magux; %neg
  elseif psi < 3*pi/2
    uz_over_ux = tan(psi - pi);
    magux = sqrt((1 - uy^2)/(1 + uz_over_ux^2));
    maguz = magux*uz_over_ux;
    uz = -maguz; %neg
    ux = -magux; %neg
  else
    uz_over_ux = tan(2*pi - psi);
    magux = sqrt((1 - uy^2)/(1 + uz_over_ux^2));
    maguz = magux*uz_over_ux;
    uz = -maguz;  %neg
    ux = magux; %pos
  end
  
  %Now find the components of v
  %Remember for MIT and CMU, v means different things  
  if strcmp(skel.type,'acclaim') || strcmp(skel.type,'bvh') || ...
      strcmp(skel.type,'cmubvh')
    %CMU DATA
    vy = -cos(phi(ii));
  elseif strcmp(skel.type,'mit')
    vy = -cos(theta(ii));
  else
    error('Unknown skeleton type');
  end 
  
  %now we have a quadratic equation to solve for vz
  a = (ux^2 + uz^2);
  b = 2*uy*uz*vy;
  c = (ux^2*vy^2 - ux^2 + uy^2*vy^2);

  %there are two solutions to the quadratic equation, and one is right
  %depending on the quadrant of psi
  %note the other solution corresponds to the vector coming out of the other
  %hip
  if psi > 3*pi/2  | psi < pi/2
    vz = (-b + sqrt(b^2 - 4*a*c))/(2*a);
  else
    vz = (-b - sqrt(b^2 - 4*a*c))/(2*a);
  end
  
  %The rotation matrix is built up a little differently for MIT and CMU  
  if strcmp(skel.type,'acclaim') || strcmp(skel.type,'bvh') || ...
      strcmp(skel.type,'cmubvh')
    %CMU DATA
    vx = (uy*cos(phi(ii)) - uz*vz) / ux;
    u = [ux uy uz]; %out right hip
    v = [vx vy vz]; %straight ahead
    
    %Occasionally I am finding that the solution for v is not exactly a
    %unit vector; this causes problems as the resulting rotation matrix 
    %isn't valid
    %So we check here and rescale when necessary
    if abs(norm(v)-1)>1E-5
      v = v/norm(v); %ensure that v is a unit vector
    end
    
    wv = -cross(u,v);
    
    if abs(norm(u)-1)>1E-5 || abs(norm(v)-1)>1E-5 || abs(norm(wv)-1)>1E-5
      error('Expected unit vectors; one of u,v,wv is not a unit vector');
    end
    %   %NOTE that the r__ subscripts refer to the order, and not to our definition
    %   %that we have used in the rest of this code
    %   %Yes -- it is confusing!
    %   %here, x- 1st dimension, y- 2nd dimension, z- 3rd dimension
    %   %regardless of what these dimensions represent
    rxx = dot(u,[1 0 0]);
    rxy = dot(u,[0 1 0]);
    rxz = dot(u,[0 0 1]);
    ryx = dot(wv, [1 0 0]);
    ryy = dot(wv, [0 1 0]);
    ryz = dot(wv, [0 0 1]);
    rzx = dot(v, [1 0 0]);
    rzy = dot(v, [0 1 0]);
    rzz = dot(v, [0 0 1]);

    R = [rxx ryx rzx; rxy ryy rzy; rxz ryz rzz]';
    %Now for display purposes, we want the exponential map representation
    %So do the conversion, based on the rotation matrix
    r = rotmat2expmap(R);
    newdata(ii,skel.tree(1).expmapInd) = r;

  elseif strcmp(skel.type,'mit')
    %MIT DATA
    vx = (uy*cos(theta(ii)) - uz*vz) / ux;
    u = [ux uz uy];
    v = [vx vz vy];
    wv = cross(u,v);

    %   %NOTE that the r__ subscripts refer to the order, and not to our definition
    %   %that we have used in the rest of this code
    %   %Yes -- it is confusing!
    %   %here, x- 1st dimension, y- 2nd dimension, z- 3rd dimension
    %   %regardless of what these dimensions represent
    rxx = dot(u,[1 0 0]);
    rxy = dot(u,[0 1 0]);
    rxz = dot(u,[0 0 1]);
    ryx = dot(v, [1 0 0]);
    ryy = dot(v, [0 1 0]);
    ryz = dot(v, [0 0 1]);
    rzx = dot(wv, [1 0 0]);
    rzy = dot(wv, [0 1 0]);
    rzz = dot(wv, [0 0 1]);
    R = [rxx ryx rzx; rxy ryy rzy; rxz ryz rzz]';
    %Now for display purposes, we want the exponential map representation
    %So do the conversion, based on the rotation matrix
    r = rotmat2expmap(R);
    %Save the Exponential map angle representation
    %NOT SURE why rotmat2expmap returns the negative
    %But the negative here seems to work
    newdata(ii,skel.tree(1).or) = -r;
  else
    error('Unknown skeleton type');
  end

end

%clear unnecessary variables
clear rxx rxy rxz ryx ryy ryz rzx rzy rzz R Eul u v wv ux uy uz vx vy vz;
clear phi theta psi a b c maguz maguz uz_over_ux;

