function Eul = RotMat2Euler(R)

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
% application.  All use of these programs is entirely at the user's own
% risk.
%
% Finds one of two equivalent Euler angle representations for a Direction
% Cosine Matrix
% Assumes the DCM is in 'zyx' order
% Given R, the rotation matrix
% Returns a vector of Euler angles (in radians)
%  the first about x axis, the second about y axis, the third about z axis
% Based on an article by Gregory G. Slabaugh
%
% Usage Eul = RotMat2Euler(R)

%Note we need to treat the case of cos(E2) = +- pi/2 separately
%This corresponds to element R(1,3) = +- 1

if R(1,3) == 1 | R(1,3) == -1
  %special case
  E3 = 0; %set arbitrarily
  dlta = atan2(R(1,2),R(1,3));
  if R(1,3) == -1
    E2 = pi/2;
    E1 = E3 + dlta;
  else
    E2 = -pi/2;
    E1 = -E3 + dlta;
  end
else
  E2 = - asin(R(1,3));
  E1 = atan2(R(2,3)/cos(E2), R(3,3)/cos(E2));
  E3 = atan2(R(1,2)/cos(E2), R(1,1)/cos(E2));
end

Eul = [E1 E2 E3];