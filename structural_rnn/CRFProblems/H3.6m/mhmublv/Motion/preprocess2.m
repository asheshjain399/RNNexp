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
% We support two types of skeletons:
%  1) Those built from the CMU database (acclaim)
%     http://mocap.cs.cmu.edu/
%  2) Those built from data from Eugene Hsu (mit)
%     http://people.csail.mit.edu/ehsu/work/sig05stf/
% This program preprocesses data (stage 2 of 2)
% We extract the dimensions of interest and form "mini-batches"
% We also scale the data.
% Certain joint angles are 1-2 DOF, so don't model constant zero cols
%

clear batchdata minibatchdata batchdataindex
batchsize = 100;        %size of minibatches
numvalidatebatches = 0; %num of minibatches to use for the validation set

clear seqlengths;

if strcmp(skel.type,'acclaim')
 %CMU-style data
 %No offsets, but several dimensions are constant 
 indx = [ 1:6 ...        %root (special representation)
   10:12 13 16:18 19 ... %lfemur ltibia lfoot ltoes
   25:27 28 31:33 34 ... %rfemur rtibia rfoot rtoes
   37:39 40:42 43:45 46:48 49:51 52:54 ... %lowerback upperback thorax lowerneck upperneck %head
   58:60 61 65 67:69 73:75 ... %(lclavicle ignored) lhumerus lradius lwrist lhand (fingers are constant) lthumb
   79:81 82 86 88:90 94:96 ];  %(rclavicle ignored) rhumerus rradius rwrist rhand (fingers are constant) rthumb    

elseif strcmp(skel.type,'mit')
  %MIT-style data
  indx = [   1:6 7:9 14 19:21 26 31:33 38 43:45 50 55:57 61:63 67:69 ...
    73:75 79:81 85:87 91:93 97:99 103:105 ];
  %Save the offsets, they will be inserted later
  offsets = [  Motion{1}(1,10:12); Motion{1}(1,16:18); ...
    Motion{1}(1,22:24); Motion{1}(1,28:30); Motion{1}(1,34:36); ...
    Motion{1}(1,40:42); Motion{1}(1,46:48); Motion{1}(1,52:54); ...
    Motion{1}(1,58:60); Motion{1}(1,64:66); Motion{1}(1,70:72); ...
    Motion{1}(1,76:78); Motion{1}(1,82:84); Motion{1}(1,88:90); ...
    Motion{1}(1,94:96); Motion{1}(1,100:102); Motion{1}(1,106:108)];
elseif strcmp(skel.type,'bvh')
  %some dimensions are constant (don't model)
  BVH_NUM_DIMS = 75;
  BVH_CONST_DIMS = [10:12 16:24 34:39 49:51 61:63 73:75];
  indx = setdiff(1:BVH_NUM_DIMS,BVH_CONST_DIMS);  
elseif strcmp(skel.type,'cmubvh')
  %some dimensions are constant (don't model)
  BVH_NUM_DIMS = 96;
  %We don't need to get rid of toes, but we do to make our space smaller
  %This means dimensions 20:21, 35:36 (19,34 are constant anyway)
  BVH_CONST_DIMS = [7:9 13 19:21 22:24 28 34:36 55:57 62 64 66 70:75 76:78 83 85 87 91:96];
  indx = setdiff(1:BVH_NUM_DIMS,BVH_CONST_DIMS); 
else
  error('Unknown skeleton type');
end

%combine the data into a large batch
batchdata = cell2mat(Motion'); %flatten it into a standard 2d array
batchdata = batchdata(:,indx);
numcases = size(batchdata,1);

%Normalize the data
data_mean = mean(batchdata,1);
data_std = std(batchdata);
batchdata =( batchdata - repmat(data_mean,numcases,1) ) ./ ...
  repmat( data_std, numcases,1);

%Index the valid cases (we don't want to mix sequences)
%This depends on the order of our model
for jj=1:length(Motion)
  seqlengths(jj) = size(Motion{jj},1);
  if jj==1 %first sequence
    batchdataindex = n1+1:seqlengths(jj);
  else
    batchdataindex = [batchdataindex batchdataindex(end)+n1+1: ...
      batchdataindex(end)+seqlengths(jj)];
  end
end

% Here's a convenient place to remove offending frames from the index
% example: offendingframes = [231 350 714 1121];
% batchdataindex = setdiff(batchdataindex,offendingframes);


%now that we know all the valid starting frames, we can randomly permute
%the order, such that we have a balanced training set
permindex = batchdataindex(randperm(length(batchdataindex)));

%should we keep some minibatches as a validation set?
numfullbatches = floor(length(permindex)/batchsize);

%fit all minibatches of size batchsize
%note that since reshape works in colums, we need to do a transpose here
minibatchindex = reshape(permindex(1: ...
  batchsize*(numfullbatches-numvalidatebatches)),...
  batchsize,numfullbatches-numvalidatebatches)';
%no need to reshape the validation set into mini-batches
%we treat it as one big batch
validatebatchindex = permindex(...
  batchsize*(numfullbatches-numvalidatebatches)+1: ...
  batchsize*numfullbatches);  
%Not all minibatches will be the same length ... must use a cell array (the
%last batch is different)
minibatch = num2cell(minibatchindex,2);
%tack on the leftover frames (smaller last batch)
leftover = permindex(batchsize*numfullbatches+1:end);
minibatch = [minibatch;num2cell(leftover,2)];

