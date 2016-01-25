function [ positions ] = getPosFromExpmap( skel_expmap,channels )
    positions = [];
    for i = 1:size(channels,1)
        position = exp2xyz(skel_expmap,channels(i,:));
        positions = [positions;position];
    end;

end

