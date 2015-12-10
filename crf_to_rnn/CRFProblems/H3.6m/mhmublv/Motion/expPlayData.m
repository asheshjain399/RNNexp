function [xlim1, ylim2, zlim3] = ...
    expPlayData(skel, channels, frameLength, writerObj, close_before_returning, xlim1, ylim2, zlim3)

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
x0=1000;
y0=1000;
width=550;
height=400;
%set(gcf,'units','points','position',[x0,y0,width,height])


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
            maxY3 = max([Y(:, 3); maxY3]);0
        end
        xlim = [minY1 maxY1];
        ylim = [minY3 maxY3];
        zlim = [minY2 maxY2];
end

set(gca, 'xlim', xlim, 'ylim', ylim, 'zlim', zlim);

% Play the motion
%writerObj = VideoWriter('generated.avi');
%writerObj.FrameRate = 100.0;
%open(writerObj);

ax = gca;
ax.XLimMode = 'manual';
ax.YLimMode = 'manual';
ax.ZLimMode = 'manual';
set(gca,'LineWidth',3,'XTickLabel',[],'YTickLabel',[],'ZTickLabel',[],'XLabel',[],'YLabel',[],'ZLabel',[]);
set(handle,'color','g');
set(handle(1),'MarkerSize',30);
set(handle(2:end),'LineWidth',5);
init_size = true;

for jj = 1:size(channels, 1)
    set(handle(1),'MarkerSize',30);
    set(handle(2:end),'LineWidth',5);
    if  jj > 50
        set(handle,'color','b')
    end;
    %title(num2str(jj));
    
    xval = channels(jj, 1);
    yval = channels(jj, 2);
    zval = channels(jj, 3);
    
    expModify(handle, channels(jj, :), skel);

    if nargin > 3
        %disp(jj)
        drawnow;
        frame1 = getframe(gca);
        %if mod(jj ,10) == 0
        %    imwrite(frame1.cdata,[num2str(jj),'.png'],'Mode','lossless');
        %end;
        %disp(size(frame1.cdata));
        if init_size
            [w h c] = size(frame1.cdata);
            init_size = false;
        end;
        writeVideo(writerObj,imresize(frame1.cdata,[w h]));
        %writeVideo(writerObj,frame1);
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
close all;


























