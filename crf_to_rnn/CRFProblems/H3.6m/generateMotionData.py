import sys
import numpy as np
import theano
import cPickle
from neuralmodels.utils import writeMatToCSV
from neuralmodels.utils import readCSVasFloat
import os
import socket as soc

base_dir = ''
if soc.gethostname() == "napoli110.stanford.edu":
	base_dir = '/scr/ashesh/h3.6m'
elif soc.gethostname() == "ashesh":
	base_dir = '.'

def unNormalizeData(normalizedData,data_mean,data_std,dimensions_to_ignore):
	T = normalizedData.shape[0]
	D = data_mean.shape[0]
	origData = np.zeros((T,D),dtype=np.float32)
	dimensions_to_use = []
	for i in range(D):
		if i in dimensions_to_ignore:
			continue
		dimensions_to_use.append(i)
	dimensions_to_use = np.array(dimensions_to_use)
	origData[:,dimensions_to_use] = normalizedData
	
	stdMat = data_std.reshape((1,D))
	stdMat = np.repeat(stdMat,T,axis=0)
	meanMat = data_mean.reshape((1,D))
	meanMat = np.repeat(meanMat,T,axis=0)
	origData = np.multiply(origData,stdMat) + meanMat
	return origData

def convertAndSave(fname):
	fpath = path_to_trajfiles + fname
	if not os.path.exists(fpath):
		return
	normalizedData = readCSVasFloat(fpath)
	origData = unNormalizeData(normalizedData,data_mean,data_std,dimensions_to_ignore)
	fpath = path_to_trajfiles + fname
	writeMatToCSV(origData,fpath)

path_to_trajfiles = '{0}/checkpoints_LSTM_no_Trans_no_rot_s_1000_lstm_init_orthogonal_fc_init_uniform_decay_type_schedule/'.format(base_dir)
numexamples = 6
epoches = 100

data_stats = cPickle.load(open('{0}h36mstats.pik'.format(path_to_trajfiles)))
data_mean = data_stats['mean']
data_std = data_stats['std']
dimensions_to_ignore = data_stats['ignore_dimensions']

for n in range(numexamples):
	fname = 'ground_truth_forecast_N_{0}'.format(n)
	convertAndSave(fname)
	fname = 'train_example_N_{0}'.format(n)
	convertAndSave(fname)
	for e in range(epoches):
		fname = 'train_error_epoch_{1}_N_{0}'.format(n,e)
		convertAndSave(fname)
		fname = 'forecast_epoch_{0}_N_{1}'.format(e,n)
		convertAndSave(fname)
