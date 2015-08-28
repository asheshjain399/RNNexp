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

% This program counts and displays the number of CRBM parameters 
% (for only the current CRBM)

fprintf('Number of current CRBM parameters:\n');
fprintf('w: %i\n',prod(size(w)));
fprintf('A: %i\n',prod(size(A)));
fprintf('B: %i\n',prod(size(B)));
fprintf('bi: %i\n',prod(size(bi)));
fprintf('bj: %i\n',prod(size(bj)));
fprintf('------------------\n');
fprintf('Total: %i\n', prod(size(w))+prod(size(A))+prod(size(B))+ ...
  prod(size(bi))+prod(size(bj)));


