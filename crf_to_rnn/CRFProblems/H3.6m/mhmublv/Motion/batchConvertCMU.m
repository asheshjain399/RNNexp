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

% For a series of sequences stored in the cell array Motion
% Go through each, and convert CMU "Euler" data to exponential maps

for jj = 1:length(Motion)
  for ii=1:size(Motion{jj},1)
    [skel1,Motion1{jj}(ii,:)] = euler2expmap(skel,Motion{jj}(ii,:));
  end
end