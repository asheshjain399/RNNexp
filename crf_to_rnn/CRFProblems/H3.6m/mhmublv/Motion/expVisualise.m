function handle = expVisualise(channels, skel, padding)

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
% Based on skelVisualise.m version 1.1
% Copyright (c) 2006 Neil D. Lawrence
%
% We support two types of skeletons:
%  1) Those built from the CMU database (acclaim)
%     http://mocap.cs.cmu.edu/
%  2) Those built from data from Eugene Hsu (mit)
%     http://people.csail.mit.edu/ehsu/work/sig05stf/
% EXPVISUALISE For updating a skel representation of 3-D data.
%
% Usage: handle = expVisualise(channels, skel, padding)

if nargin<3
    padding = 0;
end
channels = [channels zeros(1, padding)];
vals = exp2xyz(skel, channels);
connect = skelConnectionMatrix(skel);

indices = find(connect);
[I, J] = ind2sub(size(connect), indices);

handle(1) = plot3(vals(:, 1), vals(:, 3), vals(:, 2), '.');

%this is certainly not needed for MIT data
if any(strcmp(skel.type,{'acclaim','bvh','cmubvh'}))
  axis ij % make sure the left is on the left. Added in MOCAP0p134
end

set(handle(1), 'markersize', 20);
hold on
grid on
for i = 1:length(indices)
    handle(i+1) = line([vals(I(i), 1) vals(J(i), 1)], ...
        [vals(I(i), 3) vals(J(i), 3)], ...
        [vals(I(i), 2) vals(J(i), 2)]);
    set(handle(i+1), 'linewidth', 2);
end
axis equal
xlabel('x')
ylabel('z')
zlabel('y')
axis on
