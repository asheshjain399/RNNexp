%{
Takes eucliden map input which is centered w.r.t. the starting point and
convert it to w.r.t. world co-ordinate space
%}

function [channels_reconstruct,R_prev,T_prev] = revertCoordinateSpace(channels_self,R0,T0)
    addpaths;
    channels_reconstruct = channels_self;
    R_prev = R0;
    T_prev = T0;
    rootRotInd = 4:6;
    for ii = 1:size(channels_self,1)
        R = expmap2rotmat(channels_self(ii,rootRotInd))*R_prev;
        channels_reconstruct(ii,rootRotInd) = rotmat2expmap(R);
        T = T_prev + ((R_prev^-1)*(channels_self(ii,1:3))')';
        channels_reconstruct(ii,1:3) = T;
        T_prev = T;
        R_prev = R;
    end;
end