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
% manually make skel data structure for the mit data:
% http://people.csail.mit.edu/ehsu/work/sig05stf/
% based on the skel data structure from Neil Lawrence's motion capture
% toolbox: http://www.cs.man.ac.uk/~neill/mocap/  

skel.tree(1).children = [2 6 10];
 skel.tree(2).children = 3;
  skel.tree(3).children = 4;
   skel.tree(4).children = 5;
    skel.tree(5).children = [];
 skel.tree(6).children = 7;
  skel.tree(7).children = 8;
   skel.tree(8).children = 9;
    skel.tree(9).children = [];
 skel.tree(10).children = [11 15];
  skel.tree(11).children = 12;
   skel.tree(12).children = 13;
    skel.tree(13).children = 14;
     skel.tree(14).children = [];
  skel.tree(15).children = 16;
   skel.tree(16).children = 17;
    skel.tree(17).children = 18;
     skel.tree(18).children = [];
     
skel.tree(1).parent = [];
 skel.tree(2).parent = 1;
  skel.tree(3).parent = 2;
   skel.tree(4).parent = 3;
    skel.tree(5).parent = 4;
 skel.tree(6).parent = 1;
  skel.tree(7).parent = 6;
   skel.tree(8).parent = 7;
    skel.tree(9).parent = 8;
 skel.tree(10).parent = 1;
  skel.tree(11).parent = 10;
   skel.tree(12).parent = 11;
    skel.tree(13).parent = 12;
     skel.tree(14).parent = 13;
  skel.tree(15).parent = 10;
   skel.tree(16).parent = 15;
    skel.tree(17).parent = 16;
     skel.tree(18).parent = 17;
     
counter = 1;     
for jj=1:length(skel.tree)
  skel.tree(jj).or = counter:counter+2;
  skel.tree(jj).offset = counter+3:counter+5;
  counter = counter+6;
end

skel.type = 'mit';