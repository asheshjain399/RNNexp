function r=rotmat2expmap(R)

% Software provided by Hao Zhang
% http://www.cs.berkeley.edu/~nhz/software/rotations

r=quat2expmap(rotmat2quat(R));

