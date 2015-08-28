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

% This program uses a single level CRBM (or the first level of a multi-
% level CRBM) to generate data

% The program assumes that the following variables are set externally:
% numframes    -- number of frames to generate
% fr           -- a starting frame from initdata (for initialization)

numGibbs = 30; %number of alternating Gibbs iterations 

numdims = size(initdata,2);

%initialize visible layer
visible = zeros(numframes,numdims);
visible(1:n1,:) = initdata(fr:fr+n1-1,:);
%initialize hidden layer
hidden1 = ones(numframes,numhid1);

for tt=n1+1:numframes
  
  %initialize using the last frame + noise
  visible(tt,:) = visible(tt-1,:) + 0.01*randn(1,numdims);
  
  %Dynamic biases aren't re-calculated during Alternating Gibbs
  %First, add contributions from autoregressive connections 
  bistar = zeros(numdims,1);
  for hh=1:n1
    %should modify to data * A'
    bistar = bistar +  A1(:,:,hh)*visible(tt-hh,:)' ;
  end

  %Next, add contributions to hidden units from previous time steps
  bjstar = zeros(numhid1,1);
  for hh = 1:n1
    bjstar = bjstar + B1(:,:,hh)*visible(tt-hh,:)';
  end

  %Gibbs sampling
  for gg = 1:numGibbs
    %Calculate posterior probability -- hidden state being on (estimate)
    %add in bias
    bottomup =  w1*(visible(tt,:)./gsd)';
    eta = bottomup + ...                   %bottom-up connections
      bj1 + ...                            %static biases on unit
      bjstar;                              %dynamic biases
    
    hposteriors = 1./(1 + exp(-eta));      %logistic
    
    hidden1(tt,:) = double(hposteriors' > rand(1,numhid1));
    
    %Downward pass; visibles are Gaussian units
    %So find the mean of the Gaussian    
    topdown = gsd.*(hidden1(tt,:)*w1);
    
    %Mean-field approx
    visible(tt,:) = topdown + ...            %top down connections
      bi1' + ...                             %static biases
      bistar';                               %dynamic biases     
  
  end

  %If we are done Gibbs sampling, then do a mean-field sample
  %(otherwise very noisy)  
  topdown = gsd.*(hposteriors'*w1);                 

  visible(tt,:) = topdown + ...              %top down connections
      bi1' + ...                             %static biases
      bistar';                               %dynamic biases


end


  

