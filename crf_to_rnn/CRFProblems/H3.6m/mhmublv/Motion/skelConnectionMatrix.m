function connection = skelConnectionMatrix(skel);

% SKELCONNECTIONMATRIX Compute the connection matrix for the structure.
%
% connection = skelConnectionMatrix(skel);
%

% Copyright (c) 2006 Neil D. Lawrence
% skelConnectionMatrix.m version 1.1

connection = zeros(length(skel.tree));
for i = 1:length(skel.tree);
  for j = 1:length(skel.tree(i).children)    
    connection(i, skel.tree(i).children(j)) = 1;
  end
end

