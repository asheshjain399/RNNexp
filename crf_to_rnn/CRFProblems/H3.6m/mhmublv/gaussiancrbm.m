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

% This program trains a Conditional Restricted Boltzmann Machine in which
% visible, Gaussian-distributed inputs are connected to
% hidden, binary, stochastic feature detectors using symmetrically
% weighted connections. Learning is done with 1-step Contrastive Divergence.
% Directed connections are present, from the past nt configurations of the
% visible units to the current visible units (A), and the past nt
% configurations of the visible units to the current hidden units (B)

% The program assumes that the following variables are set externally:
% nt        -- order of the model
% gsd       -- fixed standard deviation of Gaussian visible units
% numepochs -- maximum number of epochs
% numhid    --  number of hidden units 
% batchdata --  a matrix of data (numcases,numdims) 
% minibatch -- a cell array of dimension batchsize, indexing the valid
% frames in batchdata
% restart   -- set to 1 if learning starts from beginning 


%batchdata is a big matrix of all the frames
%we index it with "minibatch", a cell array of mini-batch indices
numbatches = length(minibatch); 

numdims = size(batchdata,2); %visible dimension

%Setting learning rates
epsilonw=1e-3;  %undirected
epsilonbi=1e-3; %visibles
epsilonbj=1e-3; %hidden units
epsilonA=1e-3;  %autoregressive
epsilonB=1e-3;  %prev visibles to hidden

wdecay = 0.0002; %currently we use the same weight decay for w, A, B
mom = 0.9;       %momentum used only after 5 epochs of training

if restart==1,  
  restart=0;
  epoch=1;
  
  %Randomly initialize weights
  w = 0.01*randn(numhid,numdims);
  bi = 0.01*randn(numdims,1);
  bj = -1+0.01*randn(numhid,1); %set to favor units being "off"
  
  %The autoregressive weights; A(:,:,j) is the weight from t-j to the vis
  A = 0.01*randn(numdims, numdims, nt);
 
  %The weights from previous time-steps to the hiddens; B(:,:,j) is the
  %weight from t-j to the hidden layer
  B = 0.01*randn(numhid, numdims, nt);
  
  clear wgrad bigrad bjgrad Agrad Bgrad
  clear negwgrad negbigrad negbjgrad negAgrad negBgrad
  
  %keep previous updates around for momentum
  wupdate = zeros(size(w));
  biupdate = zeros(size(bi));
  bjupdate = zeros(size(bj));
  Aupdate = zeros(size(A));
  Bupdate = zeros(size(B));
    
end

%Main loop
for epoch = epoch:numepochs,
  errsum=0; %keep a running total of the difference between data and recon
  for batch = 1:numbatches,     
   
%%%%%%%%% START POSITIVE PHASE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%      
    numcases = length(minibatch{batch});
    mb = minibatch{batch}; %caches the indices   

    %data is a nt+1-d array with current and delayed data
    %corresponding to this mini-batch
    data = zeros(numcases,numdims,nt+1);
    data(:,:,1) = batchdata(mb,:);
    for hh=1:nt
      data(:,:,hh+1) = batchdata(mb-hh,:);
    end
     
    %Calculate contributions from directed autoregressive connections
    bistar = zeros(numdims,numcases); 
    for hh=1:nt       
      bistar = bistar +  A(:,:,hh)*data(:,:,hh+1)' ;
    end   
    
    %Calculate contributions from directed visible-to-hidden connections
    bjstar = zeros(numhid,numcases);
    for hh = 1:nt
      bjstar = bjstar + B(:,:,hh)*data(:,:,hh+1)';
    end
      
    %Calculate "posterior" probability -- hidden state being on 
    %Note that it isn't a true posterior   
    eta =  w*(data(:,:,1)./gsd)' + ...   %bottom-up connections
      repmat(bj, 1, numcases) + ...      %static biases on unit
      bjstar;                            %dynamic biases
    
    hposteriors = 1./(1 + exp(-eta));    %logistic
    
    %Activate the hidden units    
    hidstates = double(hposteriors' > rand(numcases,numhid)); 
    
    %Calculate positive gradients (note w.r.t. neg energy)
    wgrad = hidstates'*(data(:,:,1)./gsd);
    bigrad = sum(data(:,:,1)' - ...
      repmat(bi,1,numcases) - bistar,2)./gsd^2;
    bjgrad = sum(hidstates,1)';
           
    for hh=1:nt      
      Agrad(:,:,hh) = (data(:,:,1)' -  ...
        repmat(bi,1,numcases) - bistar)./gsd^2 * data(:,:,hh+1);
      Bgrad(:,:,hh) = hidstates'*data(:,:,hh+1);      
    end
    
%%%%%%%%% END OF POSITIVE PHASE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 
    %Activate the visible units
    %Find the mean of the Gaussian
    topdown = gsd.*(hidstates*w);

    %This is the mean of the Gaussian 
    %Instead of properly sampling, negdata is just the mean
    %If we want to sample from the Gaussian, we would add in
    %gsd.*randn(numcases,numdims);
    negdata =  topdown + ...            %top down connections
        repmat(bi',numcases,1) + ...    %static biases
        bistar';                        %dynamic biases

    %Now conditional on negdata, calculate "posterior" probability
    %for hiddens
    eta =  w*(negdata./gsd)' + ...     %bottom-up connections
        repmat(bj, 1, numcases) + ...  %static biases on unit (no change)
        bjstar;                        %dynamic biases (no change)

    hposteriors = 1./(1 + exp(-eta));   %logistic
    
    %Calculate negative gradients
    negwgrad = hposteriors*(negdata./gsd); %not using activations
    negbigrad = sum( negdata' - ...
      repmat(bi,1,numcases) - bistar,2)./gsd^2;
    negbjgrad = sum(hposteriors,2);
    
    for hh=1:nt
      negAgrad(:,:,hh) = (negdata' -  ...
       repmat(bi,1,numcases) - bistar)./gsd^2 * data(:,:,hh+1);
      negBgrad(:,:,hh) = hposteriors*data(:,:,hh+1);        
    end

%%%%%%%%% END NEGATIVE PHASE  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
    
    err= sum(sum( (data(:,:,1)-negdata).^2 ));
    errsum = err + errsum;
    
    if epoch > 5 %use momentum
        momentum=mom;
    else %no momentum
        momentum=0;
    end
    
%%%%%%%%% UPDATE WEIGHTS AND BIASES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%        
    
    wupdate =  momentum*wupdate + epsilonw* ...
        ( (wgrad - negwgrad)/numcases - wdecay*w);
    biupdate = momentum*biupdate + ...
        (epsilonbi/numcases)*(bigrad - negbigrad);
    bjupdate = momentum*bjupdate + ...
        (epsilonbj/numcases)*(bjgrad - negbjgrad);

    for hh=1:nt
        Aupdate(:,:,hh) = momentum*Aupdate(:,:,hh) + ...
            epsilonA* ( (Agrad(:,:,hh) - negAgrad(:,:,hh))/numcases - ...
            wdecay*A(:,:,hh));

        Bupdate(:,:,hh) = momentum*Bupdate(:,:,hh) + ...
            epsilonB* ( (Bgrad(:,:,hh) - negBgrad(:,:,hh))/numcases - ...
            wdecay*B(:,:,hh));
    end

    w = w +  wupdate;
    bi = bi + biupdate;
    bj = bj + bjupdate;

    for hh=1:nt
        A(:,:,hh) = A(:,:,hh) + Aupdate(:,:,hh);
        B(:,:,hh) = B(:,:,hh) + Bupdate(:,:,hh);
    end
    
%%%%%%%%%%%%%%%% END OF UPDATES  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
  
  end
    
  %every 10 epochs, show output
  if mod(epoch,10) ==0
      fprintf(1, 'epoch %4i error %6.1f  \n', epoch, errsum); 
      %Could see a plot of the weights every 10 epochs
      %figure(3); weightreport
      %drawnow;
  end
    
end
    
