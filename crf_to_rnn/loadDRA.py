import sys
import numpy as np
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

'''Loads H3.6m dataset'''
sys.path.insert(0,'CRFProblems/H3.6m')
import processdata as poseDataset
poseDataset.T = 150
poseDataset.delta_shift = 100
poseDataset.runall()
print '**** H3.6m Loaded ****'

#trX,trY = poseDataset.getMalikFeatures()
#trX_forecasting,trY_forecasting = poseDataset.getMalikTrajectoryForecasting()
crf_file = './CRFProblems/H3.6m/crf'
[nodeNames,nodeList,nodeFeatureLength,nodeConnections,edgeList,edgeFeatures,nodeToEdgeConnections,trX,trY,trX_validation,trY_validation,trX_forecasting,trY_forecasting] = graph.readCRFgraph(crf_file,poseDataset)

base_dir = poseDataset.base_dir
path = '{0}/checkpoints_dra_T_150_batch_size_100_tg_100_ls_512_fc_256_initial_lr_0.001_clipnorm_25.0_noise_schd_[500.0,1000.0,1300.0,2000.0,2500.0,3300.0,4000.0,4500.0]_noise_rate_[0.1,0.3,0.5,0.7,0.8,0.9,1.0,1.2]_decay_schd_[1000.0,4000.0]_decay_rate_[0.1,0.1]/'.format(base_dir)
#path_to_checkpoints = '{0}/checkpoints_LSTM_no_Trans_s_1000_lstm_init_orthogonal_fc_init_uniform_decay_type_schedule/'.format(base_dir)
#path_to_checkpoint = '{0}/checkpoints_LSTM_no_Trans_s_1000_lstm_init_orthogonal_fc_init_uniform_decay_type_schedule/checkpoint.30'.format(base_dir)
epoch = 20
path_to_checkpoint = '{0}checkpoint.{1}'.format(path,epoch)

rnn = loadDRA(path_to_checkpoint)

forecasted_motion = rnn.predict_sequence(trX_forecasting,sequence_length=trY_forecasting.shape[0])
skel_err = np.mean(np.sqrt(np.sum(np.square((forecasted_motion - trY_forecasting)),axis=2)),axis=1)
err_per_dof = skel_err / trY_forecasting.shape[2]
print skel_err
print err_per_dof
fname = 'forecast_error_epoch_{0}'.format(epoch)
#rnn.saveForecastError(skel_err,err_per_dof,path,fname)
fname = 'forecast_epoch_{0}'.format(epoch)
rnn.saveForecastedMotion(forecasted_motion,path,fname)
print 'done'



#prediction = rnn.predict(trX[:,:3,:],1e-5)
#print '** Saving prediction **'
#rnn.saveForecastedMotion(prediction,path_to_checkpoints,epoch='prediction')
#print '** Saving action motion **'
#rnn.saveForecastedMotion(trY[:,:3,:],path_to_checkpoints,epoch='original')
