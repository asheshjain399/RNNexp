function [r]=quat2expmap(q)

% Software provided by Hao Zhang
% http://www.cs.berkeley.edu/~nhz/software/rotations
%
% function [r]=quat2expmap(q)
% convert quaternion q into exponential map r
% 
% denote the axis of rotation by unit vector r0, the angle by theta
% q is of the form (cos(theta/2), r0*sin(theta/2))
% r is of the form r0*theta
  
  if (abs(norm(q)-1)>1E-3)
    error('quat2expmap: input quaternion is not norm 1');
  end
  sinhalftheta=norm(q(2:4));
  coshalftheta=q(1);
  r0=q(2:4)/(norm(q(2:4))+eps);
  theta=2*atan2(sinhalftheta,coshalftheta);
  theta=mod(theta+2*pi,2*pi);
  %if (theta>pi), theta=2*pi-theta; r0=-r0; end
  if (theta>pi)
    theta=2*pi-theta; 
    r0=-r0; 
  end
  r=r0*theta;
