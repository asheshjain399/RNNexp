function [ skel_expmap,expmapchannels ] = getExpmapFromSkeleton( skel,channels )
    expmapchannels = [];
    for i = 1:size(channels,1)
        [skel_expmap, channel] = euler2expmap(skel,channels(i,:));
        expmapchannels = [expmapchannels;channel];
    end;
end

