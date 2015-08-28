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

% This program profiles the weights in the model currently being trained

%undirected weights
subplot(3,nt,1);
hist(w(:),50);
figtitle = sprintf('w, max: %2.4f, min: %2.4f', max(w(:)),min(w(:)));
title(figtitle);

%visible biases
subplot(3,nt,2);
hist(bi,10);
figtitle = sprintf('bi, max: %2.4f, min: %2.4f', max(bi),min(bi));
title(figtitle);

%hidden biases
subplot(3,nt,3);
hist(bj,10);
figtitle = sprintf('bj, max: %2.4f, min: %2.4f', max(bj),min(bj));
title(figtitle);

%Autoregressive weights
for ii=1:nt
  subplot(3,nt,nt+ii);
  tempm = A(:,:,ii);
  hist(tempm(:),50);
  figtitle = sprintf('A t-%i, max: %2.4f, min: %2.4f', ...
    ii,max(tempm(:)),min(tempm(:)));
  title(figtitle);
end

%Visible to hidden
for ii=1:nt
  subplot(3,nt,2*nt+ii);
  tempm = B(:,:,ii);
  hist(tempm(:),50);
  figtitle = sprintf('B t-%i, max: %2.4f, min: %2.4f', ...
    ii,max(tempm(:)),min(tempm(:)));
  title(figtitle);
end