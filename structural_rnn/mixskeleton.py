import sys
try:
	sys.path.remove('/usr/local/lib/python2.7/dist-packages/Theano-0.6.0-py2.7.egg')
except:
	print 'Theano 0.6.0 version not found'

import numpy as np
import argparse
import theano
import os
from theano import tensor as T
from neuralmodels.utils import permute 
from neuralmodels.loadcheckpoint import *
from neuralmodels.costs import softmax_loss, euclidean_loss
from neuralmodels.models import * 
from neuralmodels.predictions import OutputMaxProb, OutputSampleFromDiscrete
from neuralmodels.layers import * 
from neuralmodels.updates import Adagrad,RMSprop,Momentum,Adadelta
import cPickle
import pdb
import socket as soc
import copy
import readCRFgraph as graph
import time
from unNormalizeData import unNormalizeData
from convertToSingleVec import convertToSingleVec 

global rng
rng = np.random.RandomState(1234567890)

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--checkpoint',type=str,default='checkpoint')
parser.add_argument('--checkpoint2',type=str,default='checkpoint')
parser.add_argument('--forecast',type=str,default='malik')
parser.add_argument('--iteration',type=int,default=4500)
parser.add_argument('--iteration2',type=int,default=4500)
parser.add_argument('--motion_prefix',type=int,default=50)
parser.add_argument('--motion_suffix',type=int,default=100)
parser.add_argument('--temporal_features',type=int,default=0)
parser.add_argument('--full_skeleton',type=int,default=1)
parser.add_argument('--dataset_prefix',type=str,default='')
parser.add_argument('--train_for',type=str,default='final')
parser.add_argument('--drop_features',type=int,default=0)
parser.add_argument('--drop_id',type=int,default=9)
parser.add_argument('--swap_edgernn',nargs='*')
parser.add_argument('--swap_nodernn',nargs='*')
parser.add_argument('--swap_edgernn2',nargs='*')
parser.add_argument('--swap_nodernn2',nargs='*')
parser.add_argument('--subsample_data',type=int,default=1)
parser.add_argument('--crf',type=str,default='')

args = parser.parse_args()


'''Loads H3.6m dataset'''
print 'Loading H3.6m'
sys.path.insert(0,'CRFProblems/H3.6m')
import processdata as poseDataset
poseDataset.T = 150
poseDataset.delta_shift = 100
poseDataset.num_forecast_examples = 24
poseDataset.motion_prefix = args.motion_prefix
poseDataset.motion_suffix = args.motion_suffix
poseDataset.temporal_features = args.temporal_features
poseDataset.full_skeleton = args.full_skeleton
poseDataset.dataset_prefix = args.dataset_prefix
poseDataset.crf_file = './CRFProblems/H3.6m/crf' + args.crf
poseDataset.train_for = args.train_for
poseDataset.drop_features = args.drop_features
poseDataset.drop_id = [args.drop_id]
poseDataset.subsample_data = args.subsample_data
poseDataset.runall()
print '**** H3.6m Loaded ****'

swap_edgernn = args.swap_edgernn
swap_nodernn = args.swap_nodernn
swap_edgernn2 = args.swap_edgernn2
swap_nodernn2 = args.swap_nodernn2
iteration = args.iteration
iteration2 = args.iteration2
new_idx = poseDataset.new_idx
featureRange = poseDataset.nodeFeaturesRanges
base_dir = poseDataset.base_dir
path = '{0}/{1}/'.format(base_dir,args.checkpoint)
path2 = '{0}/{1}/'.format(base_dir,args.checkpoint2)
if not os.path.exists(path):
	print 'Checkpoint path does not exist. Exiting!!'
	sys.exit()
	
crf_file = './CRFProblems/H3.6m/crf'

if args.forecast == 'dra':
	path_to_checkpoint = '{0}checkpoint.{1}'.format(path,iteration)
	path_to_checkpoint2 = '{0}checkpoint.{1}'.format(path2,iteration2)
	if os.path.exists(path_to_checkpoint) and os.path.exists(path_to_checkpoint2) :
		[nodeNames,nodeList,nodeFeatureLength,nodeConnections,edgeList,edgeListComplete,edgeFeatures,nodeToEdgeConnections,trX,trY,trX_validation,trY_validation,trX_forecasting,trY_forecasting,trX_forecast_nodeFeatures] = graph.readCRFgraph(poseDataset,noise=0.7,forecast_on_noisy_features=True)
		print trX_forecast_nodeFeatures.keys()
		print 'Loading the model'
		model = loadmultipleDRA(path_to_checkpoint,path_to_checkpoint2,swap_edgernn,swap_edgernn2,swap_nodernn,swap_nodernn2)
		print 'Loaded DRA: ',path_to_checkpoint
		t0 = time.time()

		trY_forecasting = model.convertToSingleVec(trY_forecasting,new_idx,featureRange)
		fname = 'ground_truth_mixforecast'
		model.saveForecastedMotion(trY_forecasting,path,fname)

		trX_forecast_nodeFeatures_ = model.convertToSingleVec(trX_forecast_nodeFeatures,new_idx,featureRange)
		fname = 'motionprefixmix'
		model.saveForecastedMotion(trX_forecast_nodeFeatures_,path,fname)

		forecasted_motion = model.predict_sequence(trX_forecasting,trX_forecast_nodeFeatures,sequence_length=trY_forecasting.shape[0],poseDataset=poseDataset,graph=graph)
		forecasted_motion = model.convertToSingleVec(forecasted_motion,new_idx,featureRange)
		fname = 'forecast_iterationmix_{0}'.format(iteration)
		model.saveForecastedMotion(forecasted_motion,path,fname)

		skel_err = np.mean(np.sqrt(np.sum(np.square((forecasted_motion - trY_forecasting)),axis=2)),axis=1)
		err_per_dof = skel_err / trY_forecasting.shape[2]
		fname = 'forecast_error_iterationmix_{0}'.format(iteration)
		model.saveForecastError(skel_err,err_per_dof,path,fname)
		t1 = time.time()
		del model

