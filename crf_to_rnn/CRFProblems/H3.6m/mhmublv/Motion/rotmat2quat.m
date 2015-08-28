% Software provided by Hao Zhang
% http://www.cs.berkeley.edu/~nhz/software/rotations

function q=rotmat2quat(R)
% function q=rotmat2quat(R)
% convert a rotation matrix R into unit quaternion q
%  
% denote the axis of rotation by unit vector r0, the angle by theta
% q is of the form (cos(theta/2), r0*sin(theta/2))

  %if (norm(R*R'-eye(3,3))>1E-10 || det(R)<0),
  %  error('rotmat2quat: input matrix is not a rotation matrix');
  %end
  d=R-R';
  r(1)=-d(2,3);
  r(2)=d(1,3);
  r(3)=-d(1,2);
  sintheta=norm(r)/2;
  r0=r/(norm(r)+eps);
  costheta=(trace(R)-1)/2;
  
  theta=atan2(sintheta,costheta);
  q=[cos(theta/2) r0*sin(theta/2)];
