import sys
import numpy as np
import theano
import cPickle
from neuralmodels.utils import writeMatToCSV
from neuralmodels.utils import readCSVasFloat
import os
import socket as soc

base_dir = open('basedir','r').readline().strip()

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

	if not len(dimensions_to_use) == normalizedData.shape[1]:
		return []
		
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
		return False
	normalizedData = readCSVasFloat(fpath)
	origData = unNormalizeData(normalizedData,data_mean,data_std,dimensions_to_ignore)
	if len(origData) > 0:
		fpath = path_to_trajfiles + fname
		writeMatToCSV(origData,fpath)
		return True

all_checkpoints = [
		'checkpoints_dra_T_150_bs_100_tg_100_ls_512_fc_256_initial_lr_0.001_clipnorm_25.0_nschd_[250,500.0,1000.0,1300.0,2000.0,2500.0,3300.0]_nrate_[0.01,0.05,0.1,0.2,0.3,0.5,0.7]_decschd_[1500.0,4500.0]_decrate_[0.1,0.1]_fc_fs_final',
		'checkpoints_dra_T_150_bs_100_tg_100_ls_512_fc_256_initial_lr_0.001_clipnorm_25.0_nschd_[250,500.0,1000.0,1300.0,2000.0,2500.0,3300.0]_nrate_[0.01,0.05,0.1,0.2,0.3,0.5,0.7]_decschd_[1500.0,4500.0]_decrate_[0.1,0.1]_fs_final'
		]

numexamples = 25 # This is the number of test mocap sequences
upto_iterations = 2000 # This is iteration upto which you want to generate motion

for checkpoint_dir in all_checkpoints:

	path_to_trajfiles = '{0}/{1}/'.format(base_dir,checkpoint_dir)

	if not os.path.exists(path_to_trajfiles):
		continue

	
	data_stats = cPickle.load(open('{0}h36mstats.pik'.format(path_to_trajfiles)))
	data_mean = data_stats['mean']
	data_std = data_stats['std']
	dimensions_to_ignore = data_stats['ignore_dimensions']
	
	for n in range(numexamples):
		fname = 'ground_truth_forecast_N_{0}'.format(n)
		convertAndSave(fname)	
		fpath = path_to_trajfiles + fname
		print fpath
		fname = 'motionprefix_N_{0}'.format(n)
		convertAndSave(fname)	
		fpath = path_to_trajfiles + fname
		
		for e in range(upto_iterations):
			fname = 'forecast_epoch_{0}_N_{1}'.format(e,n)
			convertAndSave(fname)	
			fpath = path_to_trajfiles + fname
		for iterations in range(0,5500,250):
			fname = 'forecast_iteration_{0}_N_{1}'.format(iterations,n)
			convertAndSave(fname)	
			fpath = path_to_trajfiles + fname

