function xyz = exp2xyz(skel,channels)

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
% Based on acclaim2xyz.m version 1.1
% Copyright (c) 2006 Neil D. Lawrence
%
% We support two types of skeletons:
%  1) Those built from the CMU database (acclaim)
%     http://mocap.cs.cmu.edu/
%  2) Those built from data from Eugene Hsu (mit)
%     http://people.csail.mit.edu/ehsu/work/sig05stf/
%     
% EXP2XYZ Compute XYZ values given skeleton structure and channels.
%
% Usage: xyz = exp2xyz(skel, channels)
%

if strcmp(skel.type,'acclaim')
    %CMU DATA
    rotVal = channels(skel.tree(1).expmapInd);
    xyzStruct(1).rot = expmap2rotmat(rotVal);

    xyzStruct(1).xyz = skel.tree(1).offset;
    for ii = 1:length(skel.tree(1).posInd)
        pind = skel.tree(1).posInd(ii);
        if pind
            xyzStruct(1).xyz(ii) = xyzStruct(1).xyz(ii) + channels(pind);
        end
    end

elseif strcmp(skel.type,'mit')
    %MIT DATA
    rotVal = channels(skel.tree(1).or);
    xyzStruct(1).rot = expmap2rotmat(rotVal)';

    xyzStruct(1).xyz = channels(skel.tree(1).offset);
    
elseif strcmp(skel.type,'bvh') || strcmp(skel.type,'cmubvh')
  %modeled on bvh2xyz
  for ii = 1:length(skel.tree)
    if ~isempty(skel.tree(ii).posInd)
      xpos = channels(skel.tree(ii).posInd(1));
      ypos = channels(skel.tree(ii).posInd(2));
      zpos = channels(skel.tree(ii).posInd(3));
    else
      xpos = 0;
      ypos = 0;
      zpos = 0;
    end
    xyzStruct(ii) = struct('rotation', [], 'xyz', []);
    if nargin < 2 | isempty(skel.tree(ii).expmapInd)
      r = [0 0 0];
    else            
      r = channels(skel.tree(ii).expmapInd);      
    end
    thisRotation = expmap2rotmat(r);
    %thisRotation = rotationMatrix(xangle, yangle, zangle, skel.tree(ii).order);
    thisPosition = [xpos ypos zpos];
    if ~skel.tree(ii).parent
      xyzStruct(ii).rotation = thisRotation;
      xyzStruct(ii).xyz = skel.tree(ii).offset + thisPosition;
    else
      xyzStruct(ii).xyz = ...
        (skel.tree(ii).offset + thisPosition)*xyzStruct(skel.tree(ii).parent).rotation ...
        + xyzStruct(skel.tree(ii).parent).xyz;
      xyzStruct(ii).rotation = thisRotation*xyzStruct(skel.tree(ii).parent).rotation;

    end
  end
        
else
    error('Unknown skeleton type');
end

if strcmp(skel.type,'acclaim') || strcmp(skel.type,'mit')
  %CMU & MIT process the root first, then proceed through tree  
  
  for ii = 1:length(skel.tree(1).children)
    ind = skel.tree(1).children(ii);
    xyzStruct = getChildXyz(skel, xyzStruct, ind, channels);
  end
end

xyz = reshape([xyzStruct(:).xyz], 3, length(skel.tree))';

if strcmp(skel.type,'mit')
    %Note that for the CMU and MIT representations, the 2nd and 3rd xyz
    %dimensions are reversed; If we commit to the CMU representation now,
    %this will eliminate having to treat them differently later on
    xyz = [xyz(:,1) xyz(:,3) xyz(:,2)];
end

function xyzStruct = getChildXyz(skel, xyzStruct, ind, channels)
% GETCHILDXYZ

if strcmp(skel.type,'acclaim') 
    %CMU DATA
    parent = skel.tree(ind).parent;
    children = skel.tree(ind).children;

    rotVal = channels(skel.tree(ind).expmapInd);
    tdof = expmap2rotmat(rotVal);

    torient = rotationMatrix(deg2rad(skel.tree(ind).axis(1)), ...
        deg2rad(skel.tree(ind).axis(2)), ...
        deg2rad(skel.tree(ind).axis(3)), ...
        skel.tree(ind).axisOrder);
    torientInv = rotationMatrix(deg2rad(-skel.tree(ind).axis(1)), ...
        deg2rad(-skel.tree(ind).axis(2)), ...
        deg2rad(-skel.tree(ind).axis(3)), ...
        skel.tree(ind).axisOrder(end:-1:1));
    xyzStruct(ind).rot = torientInv*tdof*torient*xyzStruct(parent).rot;
    xyzStruct(ind).xyz = xyzStruct(parent).xyz + (skel.tree(ind).offset*xyzStruct(ind).rot);

elseif strcmp(skel.type,'mit')
    %MIT DATA
    parent = skel.tree(ind).parent;
    children = skel.tree(ind).children;
    tdof = expmap2rotmat(channels(skel.tree(ind).or))';
    xyzStruct(ind).rot = tdof*xyzStruct(parent).rot;
   
    xyzStruct(ind).xyz = xyzStruct(parent).xyz + channels(skel.tree(ind).offset)*xyzStruct(parent).rot;
else
    error('Unknown skeleton type');
end

for ii = 1:length(children)
    cind = children(ii);
    xyzStruct = getChildXyz(skel, xyzStruct, cind, channels);
end

function theta = deg2rad(omega)
% DEG2RAD
% Originally was supplied by Neil Lawrence's NDLUTIL toolbox
% But we just provide it locally here to remove the dependency

theta = omega/180*pi;
