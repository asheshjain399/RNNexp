function R=expmap2rotmat(r);

% Software provided by Hao Zhang
% http://www.cs.berkeley.edu/~nhz/software/rotations
%
% function R=expmap2rotmat(r);
% convert exponential map r into a rotation matrix R
%
% denote the axis of rotation by unit vector r0, the angle by theta
% r is of the form r0*theta 
  
  theta=norm(r);
  r0=r/(norm(r)+eps);
 %if (theta>pi) 
 %  warning('expmap2rotmat: exp map rotation angle > pi, not in canonical form');
 %end
  r0x=[0 -r0(3) r0(2);0 0 -r0(1); 0 0 0];
  r0x=r0x-r0x';
  R=eye(3,3)+sin(theta)*r0x+(1-cos(theta))*r0x*r0x;
  