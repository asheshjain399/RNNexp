function [xlim, ylim, zlim] = ...
    expPlayData(skel, channels, frameLength, writerObj, close_before_returning, xlim, ylim, zlim)

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
% Based on skelPlayData.m version 1.1
% Copyright (c) 2006 Neil D. Lawrence
%
% We support two types of skeletons:
%  1) Those built from the CMU database (acclaim)
%     http://mocap.cs.cmu.edu/
%  2) Those built from data from Eugene Hsu (mit)
%     http://people.csail.mit.edu/ehsu/work/sig05stf/
% EXPPLAYDATA Play skel motion capture data.
% Data is in exponential map representation
%
% Usage: [xlim, ylim, zlim] = expPlayData(skel, channels, frameLength)

if nargin < 3
    frameLength = 1/120;
end
%set(gcf,'Visible','off');
clf
%set(gcf,'Visible','off');
%figure;
handle = expVisualise(channels(1, :), skel);

if nargin < 8
    %We didn't specify the limits of the motion
    %So calculate the limits

        xlim = get(gca, 'xlim');
        minY1 = xlim(1);
        maxY1 = xlim(2);
        ylim = get(gca, 'ylim');
        minY3 = ylim(1);
        maxY3 = ylim(2);
        zlim = get(gca, 'zlim');
        minY2 = zlim(1);
        maxY2 = zlim(2);
        for ii = 1:size(channels, 1)
            Y = exp2xyz(skel, channels(ii, :));
            minY1 = min([Y(:, 1); minY1]);
            minY2 = min([Y(:, 2); minY2]);
            minY3 = min([Y(:, 3); minY3]);
            maxY1 = max([Y(:, 1); maxY1]);
            maxY2 = max([Y(:, 2); maxY2]);
            maxY3 = max([Y(:, 3); maxY3]);
        end
        xlim = [minY1 maxY1];
        ylim = [minY3 maxY3];
        zlim = [minY2 maxY2];
end

set(gca, 'xlim', xlim, ...
    'ylim', ylim, ...
    'zlim', zlim);

% Play the motion
%writerObj = VideoWriter('generated.avi');
%writerObj.FrameRate = 100.0;
%open(writerObj);

for jj = 1:size(channels, 1)
    if  jj > 50
        set(handle(1),'color','g')
    end;
    expModify(handle, channels(jj, :), skel);
    if nargin > 3
        %disp(jj)
        writeVideo(writerObj,getframe);
    else
        pause(frameLength);
    end;
end
if nargin == 4
    close(writerObj);
end;
if nargin == 5 && close_before_returning
    close(writerObj);
end;