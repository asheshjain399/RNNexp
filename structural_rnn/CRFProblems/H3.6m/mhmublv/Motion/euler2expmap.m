function [skel, expmapchannels] = euler2expmap(skel,eulerchannels)

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
% This is a function for converting from Euler CMU data to exponential maps
% It assumes that Neil Lawrence's MOCAP toolbox has been used to create a 
% skeleton structure "skel" and matrix of channels "eulerchannels".
% This function goes hierarchically through the skeleton and converts all 
% Euler angles to the exponential map representation
%
% Usage: [skel, expmapchannels] = euler2expmap(skel,eulerchannels)

%Process root
%don't touch the global translation
expmapchannels(skel.tree(1).posInd) = eulerchannels(skel.tree(1).posInd);

%keep a counter to keep track of exponential map dimensions that we are
%writing
counter = length(expmapchannels)+1;

rotVal = zeros(1, 3); %skel.tree(1).orientation;
for ii = 1:length(skel.tree(1).rotInd)
    rind = skel.tree(1).rotInd(ii);
    if rind
        rotVal(ii) = rotVal(ii) + eulerchannels(rind);
    end
end

R = rotationMatrix(deg2rad(rotVal(1)), ...
    deg2rad(rotVal(2)), ...
    deg2rad(rotVal(3)), ...
    skel.tree(1).axisOrder);

r = rotmat2expmap(R);

%write out the exponential map dimensions
expmapchannels(counter:counter+2) = r;
skel.tree(1).expmapInd = counter:counter+2;
counter = counter+3;

%Now go through children and convert their rotations to exponential maps
for i = 1:length(skel.tree(1).children)
    ind = skel.tree(1).children(i);
    [expmapchannels, skel, counter] = convertChild(skel, expmapchannels, ind, eulerchannels, counter);
end

function [expmapchannels, skel, counter] = convertChild(skel, expmapchannels, ind, eulerchannels, counter)

%Converts a child in a hierarchy to expmap representation
parent = skel.tree(ind).parent;
children = skel.tree(ind).children;
rotVal = zeros(1, 3);
for j = 1:length(skel.tree(ind).rotInd)
    rind = skel.tree(ind).rotInd(j);
    if rind
        rotVal(j) = eulerchannels(rind);
    else
        rotVal(j) = 0;
    end
end

R = rotationMatrix(deg2rad(rotVal(1)), ...
    deg2rad(rotVal(2)), ...
    deg2rad(rotVal(3)), ...
    skel.tree(ind).order);

r = rotmat2expmap(R);

%write out the exponential map dimensions
expmapchannels(counter:counter+2) = r;
skel.tree(ind).expmapInd = counter:counter+2;
counter = counter+3;

for i = 1:length(children)
    cind = children(i);
    [expmapchannels, skel, counter] = convertChild(skel, expmapchannels, cind, eulerchannels, counter);
end

