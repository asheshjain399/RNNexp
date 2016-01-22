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

% This is the "main" demo
% It trains two CRBM models, one on top of the other, and then
% demonstrates data generation

clear all; close all;
more off;   %turn off paging

%initialize RAND,RANDN to a different state
rand('state',sum(100*clock))
randn('state',sum(100*clock))

%Our important Motion routines are in a subdirectory
addpath('./Motion')

%Load the supplied training data
%Motion is a cell array containing 3 sequences of walking motion (120fps)
%skel is struct array which describes the person's joint hierarchy
load Data/data.mat

%Downsample (to 30 fps) simply by dropping frames
%We really should low-pass-filter before dropping frames
%See Matlab's DECIMATE function
dropRate=4;
dropframes;

fprintf(1,'Preprocessing data \n');

%Run the 1st stage of pre-processing
%This converts to body-centered coordinates, and converts to ground-plane
%differences
preprocess1

%how-many timesteps do we look back for directed connections
%this is what we call the "order" of the model 
n1 = 3; %first layer
n2 = 3; %second layer
        
%Run the 2nd stage of pre-processing
%This drops the zero/constant dimensions and builds mini-batches
preprocess2
numdims = size(batchdata,2); %data (visible) dimension

%save some frames of our pre-processed data for later
%we need an initialization to generate 
initdata = batchdata(1:100,:);

%Set network properties
numhid1 = 150; numhid2 = 150; numepochs=2000;
gsd=1;          %fixed standard deviation for Gaussian units
nt = n1;        %crbm "order"
numhid=numhid1;
fprintf(1,'Training Layer 1 CRBM, order %d: %d-%d \n',nt,numdims,numhid);
restart=1;      %initialize weights
%gaussiancrbm;

load('Results/layer1.mat')
w = w1; bj = bj1; bi = bi1; A = A1; B = B1;

%Plot a representation of the weights
hdl = figure(3); weightreport
set(hdl,'Name','Layer 1 CRBM weights');

%w1 = w; bj1 = bj; bi1 = bi; A1 = A; B1 = B;
%save Results/layer1.mat w1 bj1 bi1 A1 B1

getfilteringdist;
numhid = numhid2; nt=n2;
batchdata = filteringdist;
numdims = size(batchdata,2); %data (visible) dimension
fprintf(1,'Training Layer 2 CRBM, order %d: %d-%d \n',nt,numdims,numhid);
restart=1;      %initialize weights
%binarycrbm;

load('Results/layer2.mat')
w = w2; bj = bj2; bi = bi2; A = A2; B = B2;

%w2 = w; bj2 = bj; bi2 = bi; A2 = A; B2 = B;
%save Results/layer2.mat w2 bj2 bi2 A2 B2

%Now use the 2-layer CRBM to generate a sequence of data
numframes = 400; %how many frames to generate
fr = 4;         %pick a starting frame from initdata
                 %will use max(n1,n2) frames of initialization data
fprintf(1,'Generating %d-frame sequence of data from 2-layer CRBM ... \n',numframes);
gen2;
%We must postprocess the generated data before playing
%It is in the normalized angle space
postprocess;

%Plot a representation of the weights
hdl = figure(4); weightreport
set(hdl,'Name','Layer 2 CRBM weights');

%Plot top-layer activations
figure(5); imagesc(hidden2'); colormap gray;
title('Top hidden layer, activations'); ylabel('hidden units'); xlabel('frames')
%Plot middle-layer probabilities
figure(6); imagesc(hidden1'); colormap gray;
title('First hidden layer, probabilities'); ylabel('hidden units'); xlabel('frames')

fprintf(1,'Playing generated sequence\n');
figure(2); expPlayData(skel, newdata, 1/30)

