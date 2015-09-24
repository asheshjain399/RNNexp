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
	#base_dir = '/scr/ashesh/h3.6m'
	base_dir = '/scail/scratch/group/cvgl/ashesh/h3.6m'
	gpus = 1
elif soc.gethostname() == "napoli106.stanford.edu":
	#base_dir = '/scr/ashesh/h3.6m'
	base_dir = '/scail/scratch/group/cvgl/ashesh/h3.6m'
	gpus = 2
elif soc.gethostname() == "napoli107.stanford.edu":
	#base_dir = '/scr/ashesh/h3.6m'
	base_dir = '/scail/scratch/group/cvgl/ashesh/h3.6m'
	gpus = 2
elif soc.gethostname() == "napoli105.stanford.edu":
	#base_dir = '/scr/ashesh/h3.6m'
	base_dir = '/scail/scratch/group/cvgl/ashesh/h3.6m'
	gpus = 2
elif soc.gethostname() == "napoli108.stanford.edu":
	#base_dir = '/scr/ashesh/h3.6m'
	base_dir = '/scail/scratch/group/cvgl/ashesh/h3.6m'
	gpus = 2
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

#path_to_trajfiles = '{0}/checkpoints_LSTM_no_Trans_no_rot_s_1000_lstm_init_orthogonal_fc_init_uniform_decay_type_schedule/'.format(base_dir)
#path_to_trajfiles = '{0}/checkpoints_LSTM_no_Trans_no_rot_lr_0.001_mu_0.99_gclip_25.0/'.format(base_dir)
#path_to_trajfiles = '{0}/checkpoints_LSTM_no_Trans_no_rot_lr_0.001_mu_0.99_gclip_25.0_batch_size_100_truncate_gradient_100/'.format(base_dir)
#path_to_trajfiles = '{0}/checkpoints_malik_batch_size_100_noNoise_initial_lr_0.001_clipnorm_25.0_momentum_0.9_g_clip_25.0_noise_schedule_[2,6,12,20,35,50,60,80]_noise_rate_schedule_[0.01,0.05,0.075,0.1,0.5,0.8,1.0,1.2]/'.format(base_dir)


checkpoint_dir = 'checkpoints_malik_T_150_batch_size_100_initial_lr_0.001_clipnorm_25.0_momentum_0.99_g_clip_25.0_noise_schedule_[900.0,2000.0,3000.0,4000.0,5000.0,6000.0,7000.0,8000.0]_noise_rate_schedule_[0.1,0.3,0.5,0.7,0.8,0.9,1.0,1.2]'

#checkpoint_dir = 'checkpoints_malik_T_500_batch_size_20_initial_lr_0.001_clipnorm_25.0_momentum_0.99_g_clip_25.0_noise_schedule_[900.0,2000.0,3000.0,4000.0,5000.0,6000.0,7000.0,8000.0]_noise_rate_schedule_[0.1,0.3,0.5,0.7,0.8,0.9,1.0,1.2]'

# T=20 batch_size=50
checkpoint_dir = 'checkpoints_malik_T_20_batch_size_50_initial_lr_0.001_clipnorm_25.0_momentum_0.99_g_clip_25.0_noise_schedule_[900.0,2000.0,3000.0,4000.0,5000.0,6000.0,7000.0,8000.0]_noise_rate_schedule_[0.1,0.3,0.5,0.7,0.8,0.9,1.0,1.2]'
#checkpoint_dir = 'checkpoints_malik_T_20_batch_size_50_initial_lr_0.001_clipnorm_0.0_momentum_0.99_g_clip_25.0_noise_schedule_[900.0,2000.0,3000.0,4000.0,5000.0,6000.0,7000.0,8000.0]_noise_rate_schedule_[0.1,0.3,0.5,0.7,0.8,0.9,1.0,1.2]'
#checkpoint_dir = 'checkpoints_malik_T_20_batch_size_50_initial_lr_0.0001_clipnorm_0.0_momentum_0.99_g_clip_25.0_noise_schedule_[900.0,2000.0,3000.0,4000.0,5000.0,6000.0,7000.0,8000.0]_noise_rate_schedule_[0.1,0.3,0.5,0.7,0.8,0.9,1.0,1.2]'
checkpoint_dir = 'checkpoints_malik_T_20_batch_size_50_initial_lr_0.0001_clipnorm_25.0_momentum_0.99_g_clip_25.0_noise_schedule_[900.0,2000.0,3000.0,4000.0,5000.0,6000.0,7000.0,8000.0]_noise_rate_schedule_[0.1,0.3,0.5,0.7,0.8,0.9,1.0,1.2]'


# T=100 batch_size=50
#checkpoint_dir = 'checkpoints_malik_T_100_batch_size_50_tg_50_initial_lr_0.001_clipnorm_25.0_noise_schd_[500.0,1000.0,1300.0,2000.0,2500.0,3300.0,4000.0,4500.0]_noise_rate_[0.1,0.3,0.5,0.7,0.8,0.9,1.0,1.2]_decay_schd_[1000.0,4500.0]_decay_rate_[0.1,0.1]'
checkpoint_dir = 'checkpoints_malik_T_100_batch_size_50_tg_50_initial_lr_0.001_clipnorm_5.0_noise_schd_[500.0,1000.0,1300.0,2000.0,2500.0,3300.0,4000.0,4500.0]_noise_rate_[0.1,0.3,0.5,0.7,0.8,0.9,1.0,1.2]_decay_schd_[1000.0,4500.0]_decay_rate_[0.1,0.1]'
#checkpoint_dir = 'checkpoints_malik_T_100_batch_size_50_tg_50_initial_lr_0.001_clipnorm_25.0_noise_schd_[1000.0,1500.0,1800.0,2300.0,3000.0,3800.0,4000.0,4500.0]_noise_rate_[0.1,0.3,0.5,0.7,0.8,0.9,1.0,1.2]_decay_schd_[1000.0,4500.0]_decay_rate_[0.1,0.1]'
#checkpoint_dir = 'checkpoints_malik_T_100_batch_size_50_tg_50_initial_lr_0.001_clipnorm_5.0_noise_schd_[1000.0,1500.0,1800.0,2300.0,3000.0,3800.0,4000.0,4500.0]_noise_rate_[0.1,0.3,0.5,0.7,0.8,0.9,1.0,1.2]_decay_schd_[1000.0,4500.0]_decay_rate_[0.1,0.1]'


# T=100 batch_size=10
#checkpoint_dir = 'checkpoints_malik_T_100_batch_size_10_initial_lr_0.001_clipnorm_25.0_momentum_0.99_g_clip_25.0_noise_schedule_[900.0,2000.0,3000.0,4000.0,5000.0,6000.0,7000.0,8000.0]_noise_rate_schedule_[0.1,0.3,0.5,0.7,0.8,0.9,1.0,1.2]'

'''
path_to_trajfiles = '{0}/{1}/'.format(base_dir,checkpoint_dir)

model='malik'

numexamples = 6

epoches = 2000

data_stats = cPickle.load(open('{0}h36mstats.pik'.format(path_to_trajfiles)))
data_mean = data_stats['mean']
data_std = data_stats['std']
dimensions_to_ignore = data_stats['ignore_dimensions']

for n in range(numexamples):
	fname = 'ground_truth_forecast_N_{0}'.format(n)
	convertAndSave(fname)	
	fpath = path_to_trajfiles + fname
	if os.path.exists(fpath):
		os.system('scp {0} ashesh@172.24.68.124:/home/ashesh/project/NN/RNNexp/crf_to_rnn/CRFProblems/H3.6m/dataParser/Utils/{2}/{1}.dat'.format(fpath,fname,model))
	fname = 'train_example_N_{0}'.format(n)
	convertAndSave(fname)
	for e in range(epoches):
		fname = 'train_error_epoch_{1}_N_{0}'.format(n,e)
		convertAndSave(fname)
		fname = 'forecast_epoch_{0}_N_{1}'.format(e,n)
		convertAndSave(fname)	
		fpath = path_to_trajfiles + fname
		if os.path.exists(fpath):
			os.system('scp {0} ashesh@172.24.68.124:/home/ashesh/project/NN/RNNexp/crf_to_rnn/CRFProblems/H3.6m/dataParser/Utils/{2}/{1}.dat'.format(fpath,fname,model))
'''

all_checkpoints = ['checkpoints_malik_T_150_batch_size_100_initial_lr_0.001_clipnorm_25.0_momentum_0.99_g_clip_25.0_noise_schedule_[900.0,2000.0,3000.0,4000.0,5000.0,6000.0,7000.0,8000.0]_noise_rate_schedule_[0.1,0.3,0.5,0.7,0.8,0.9,1.0,1.2]',
		'checkpoints_malik_T_500_batch_size_20_initial_lr_0.001_clipnorm_25.0_momentum_0.99_g_clip_25.0_noise_schedule_[900.0,2000.0,3000.0,4000.0,5000.0,6000.0,7000.0,8000.0]_noise_rate_schedule_[0.1,0.3,0.5,0.7,0.8,0.9,1.0,1.2]',
		'checkpoints_malik_T_20_batch_size_50_initial_lr_0.001_clipnorm_25.0_momentum_0.99_g_clip_25.0_noise_schedule_[900.0,2000.0,3000.0,4000.0,5000.0,6000.0,7000.0,8000.0]_noise_rate_schedule_[0.1,0.3,0.5,0.7,0.8,0.9,1.0,1.2]',
		'checkpoints_malik_T_100_batch_size_50_tg_50_initial_lr_0.001_clipnorm_5.0_noise_schd_[500.0,1000.0,1300.0,2000.0,2500.0,3300.0,4000.0,4500.0]_noise_rate_[0.1,0.3,0.5,0.7,0.8,0.9,1.0,1.2]_decay_schd_[1000.0,4500.0]_decay_rate_[0.1,0.1]',
		'checkpoints_malik_T_100_batch_size_50_tg_50_initial_lr_0.001_clipnorm_25.0_noise_schd_[500.0,1000.0,1300.0,2000.0,2500.0,3300.0,4000.0,4500.0]_noise_rate_[0.1,0.3,0.5,0.7,0.8,0.9,1.0,1.2]_decay_schd_[1000.0,4500.0]_decay_rate_[0.1,0.1]',
		'checkpoints_malik_T_100_batch_size_10_initial_lr_0.001_clipnorm_25.0_momentum_0.99_g_clip_25.0_noise_schedule_[900.0,2000.0,3000.0,4000.0,5000.0,6000.0,7000.0,8000.0]_noise_rate_schedule_[0.1,0.3,0.5,0.7,0.8,0.9,1.0,1.2]']


all_checkpoints = ['checkpoints_malik_T_150_batch_size_100_tg_100_initial_lr_0.001_clipnorm_5.0_noise_schd_[500.0,1000.0,1300.0,2000.0,2500.0,3300.0,4000.0,4500.0]_noise_rate_[0.1,0.3,0.5,0.7,0.8,0.9,1.0,1.2]_decay_schd_[1500.0,4500.0]_decay_rate_[0.1,0.1]',
		'checkpoints_malik_T_150_batch_size_100_tg_100_initial_lr_0.001_clipnorm_5.0_noise_schd_[1000.0,1500.0,1800.0,2300.0,3000.0,3800.0,4000.0,4500.0]_noise_rate_[0.1,0.3,0.5,0.7,0.8,0.9,1.0,1.2]_decay_schd_[1500.0,4500.0]_decay_rate_[0.1,0.1]',
		'checkpoints_malik_T_150_batch_size_100_tg_100_initial_lr_0.001_clipnorm_25.0_noise_schd_[500.0,1000.0,1300.0,2000.0,2500.0,3300.0,4000.0,4500.0]_noise_rate_[0.1,0.3,0.5,0.7,0.8,0.9,1.0,1.2]_decay_schd_[1500.0,4500.0]_decay_rate_[0.1,0.1]',
		'checkpoints_malik_T_150_batch_size_100_tg_100_initial_lr_0.001_clipnorm_25.0_noise_schd_[1000.0,1500.0,1800.0,2300.0,3000.0,3800.0,4000.0,4500.0]_noise_rate_[0.1,0.3,0.5,0.7,0.8,0.9,1.0,1.2]_decay_schd_[1500.0,4500.0]_decay_rate_[0.1,0.1]',
		'checkpoints_malik_T_150_batch_size_100_tg_100_initial_lr_0.001_clipnorm_5.0_noise_schd_[500.0,1000.0,1300.0,2000.0,2500.0,3300.0,4000.0,4500.0]_noise_rate_[0.1,0.3,0.5,0.7,0.8,0.9,1.0,1.2]_decay_schd_[1000.0,4000.0]_decay_rate_[0.1,0.1]',
		'checkpoints_malik_T_150_batch_size_100_tg_100_initial_lr_0.001_clipnorm_5.0_noise_schd_[1000.0,1500.0,1800.0,2300.0,3000.0,3800.0,4000.0,4500.0]_noise_rate_[0.1,0.3,0.5,0.7,0.8,0.9,1.0,1.2]_decay_schd_[1000.0,4000.0]_decay_rate_[0.1,0.1]',
		'checkpoints_malik_T_150_batch_size_100_tg_100_initial_lr_0.001_clipnorm_25.0_noise_schd_[500.0,1000.0,1300.0,2000.0,2500.0,3300.0,4000.0,4500.0]_noise_rate_[0.1,0.3,0.5,0.7,0.8,0.9,1.0,1.2]_decay_schd_[1000.0,4000.0]_decay_rate_[0.1,0.1]',
		'checkpoints_malik_T_150_batch_size_100_tg_100_initial_lr_0.001_clipnorm_25.0_noise_schd_[1000.0,1500.0,1800.0,2300.0,3000.0,3800.0,4000.0,4500.0]_noise_rate_[0.1,0.3,0.5,0.7,0.8,0.9,1.0,1.2]_decay_schd_[1000.0,4000.0]_decay_rate_[0.1,0.1]']

all_checkpoints = ['checkpoints_dra_T_150_bs_100_tg_100_ls_512_fc_256_initial_lr_0.001_clipnorm_5.0_nschd_[250,500.0,1000.0,1300.0,2000.0,2500.0,3300.0]_nrate_[0.01,0.05,0.1,0.2,0.3,0.5,0.7]_decschd_[1500.0,4500.0]_decrate_[0.1,0.1]',
		'checkpoints_dra_T_150_bs_100_tg_100_ls_512_fc_256_initial_lr_0.001_clipnorm_25.0_nschd_[250,500.0,1000.0,1300.0,2000.0,2500.0,3300.0]_nrate_[0.01,0.05,0.1,0.2,0.3,0.5,0.7]_decschd_[1500.0,4500.0]_decrate_[0.1,0.1]',
		#'checkpoints_dra_T_150_bs_100_tg_100_ls_512_fc_256_initial_lr_0.001_clipnorm_5.0_nschd_[250,500,750,1000,1250,1500,1750,2000,2250,2500]_nrate_[0.001,0.002,0.004,0.008,0.016,0.032,0.064,0.128,0.256,0.512]_decschd_[1000.0,4000.0]_decrate_[0.1,0.1]',
		'checkpoints_dra_T_150_bs_100_tg_100_ls_512_fc_256_initial_lr_0.001_clipnorm_25.0_nschd_[250,500,750,1000,1250,1500,1750,2000,2250,2500]_nrate_[0.001,0.002,0.004,0.008,0.016,0.032,0.064,0.128,0.256,0.512]_decschd_[1000.0,4000.0]_decrate_[0.1,0.1]'
		#'checkpoints_dra_T_150_bs_100_tg_100_ls_512_fc_256_initial_lr_0.001_clipnorm_25.0_nschd_[250,500,750,1000,1250,1500,1750,2000,2250,2500]_nrate_[0.001,0.002,0.004,0.008,0.016,0.032,0.064,0.128,0.256,0.512]_decschd_[1500.0,4500.0]_decrate_[0.1,0.1]'
		]


all_checkpoints = ['checkpoints_malik_T_150_bs_100_tg_100_initial_lr_0.001_clipnorm_25.0_nschd_[250,500.0,1000.0,1300.0,2000.0,2500.0,3300.0]_nrate_[0.01,0.05,0.1,0.2,0.3,0.5,0.7]_decschd_[1500.0,4500.0]_decrate_[0.1,0.1]_fs',
		'checkpoints_malik_T_150_bs_100_tg_100_initial_lr_0.001_clipnorm_25.0_nschd_[900.0,2000.0,3000.0,4000.0,5000.0,6000.0,7000.0,8000.0]_nrate_[0.1,0.3,0.5,0.7,0.8,0.9,1.0,1.2]_decschd_[1500.0,4500.0]_decrate_[0.1,0.1]_fs',
		'checkpoints_malik_T_150_bs_100_tg_100_initial_lr_0.001_clipnorm_25.0_nschd_[250,500.0,1000.0,1300.0,2000.0,2500.0,3300.0]_nrate_[0.01,0.05,0.1,0.2,0.3,0.5,0.7]_decschd_[1500.0,4500.0]_decrate_[0.1,0.1]_wd_0.0001_fs',
		'checkpoints_malik_T_150_bs_100_tg_100_initial_lr_0.001_clipnorm_25.0_nschd_[250,500.0,1000.0,1300.0,2000.0,2500.0,3300.0]_nrate_[0.01,0.05,0.1,0.2,0.3,0.5,0.7]_decschd_[1500.0,4500.0]_decrate_[0.1,0.1]_fc_wd_0.0005_fs'
		]
#'checkpoints_malik_T_150_bs_100_tg_100_initial_lr_0.001_clipnorm_25.0_nschd_[250,500,750,1000,1250,1500,1750,2000,2250,2500,3300,4000]_nrate_[0.001,0.002,0.004,0.008,0.016,0.032,0.064,0.128,0.256,0.512,0.65,0.7]_decschd_[1000.0,4000.0]_decrate_[0.1,0.1]',
#'checkpoints_malik_T_150_bs_100_tg_100_initial_lr_0.001_clipnorm_25.0_nschd_[250,500,750,1000,1250,1500,1750,2000,2250,2500,3300,4000]_nrate_[0.001,0.002,0.004,0.008,0.016,0.032,0.064,0.128,0.256,0.512,0.65,0.7]_decschd_[1500.0,4500.0]_decrate_[0.1,0.1]'
		
for checkpoint_dir in all_checkpoints:
	path_to_trajfiles = '{0}/{1}/'.format(base_dir,checkpoint_dir)
	print path_to_trajfiles
	if not os.path.exists(path_to_trajfiles):
		continue

	os.system('ssh ashesh@172.24.68.124 mkdir -p /home/ashesh/project/NN/RNNexp/crf_to_rnn/CRFProblems/H3.6m/dataParser/Utils/{0}'.format(checkpoint_dir))

	print 'Sending: ',checkpoint_dir
	
	fname = 'logfile'
	fpath = path_to_trajfiles + fname
	os.system('scp {0} ashesh@172.24.68.124:/home/ashesh/project/NN/RNNexp/crf_to_rnn/CRFProblems/H3.6m/dataParser/Utils/{2}/{1}.dat'.format(fpath,fname,checkpoint_dir))

	fname = 'complete_log'
	fpath = path_to_trajfiles + fname
	os.system('scp {0} ashesh@172.24.68.124:/home/ashesh/project/NN/RNNexp/crf_to_rnn/CRFProblems/H3.6m/dataParser/Utils/{2}/{1}.dat'.format(fpath,fname,checkpoint_dir))

for checkpoint_dir in all_checkpoints:

	path_to_trajfiles = '{0}/{1}/'.format(base_dir,checkpoint_dir)

	if not os.path.exists(path_to_trajfiles):
		continue

	os.system('ssh ashesh@172.24.68.124 mkdir -p /home/ashesh/project/NN/RNNexp/crf_to_rnn/CRFProblems/H3.6m/dataParser/Utils/{0}'.format(checkpoint_dir))

	print 'Sending: ',checkpoint_dir

	model='malik'

	numexamples = 25

	epoches = 2000
	
	data_stats = cPickle.load(open('{0}h36mstats.pik'.format(path_to_trajfiles)))
	data_mean = data_stats['mean']
	data_std = data_stats['std']
	dimensions_to_ignore = data_stats['ignore_dimensions']

	for n in range(numexamples):
		fname = 'ground_truth_forecast_N_{0}'.format(n)
		convertAndSave(fname)	
		fpath = path_to_trajfiles + fname
		if os.path.exists(fpath):
			os.system('scp {0} ashesh@172.24.68.124:/home/ashesh/project/NN/RNNexp/crf_to_rnn/CRFProblems/H3.6m/dataParser/Utils/{2}/{1}.dat'.format(fpath,fname,checkpoint_dir))
		fname = 'motionprefix_N_{0}'.format(n)
		convertAndSave(fname)	
		fpath = path_to_trajfiles + fname
		if os.path.exists(fpath):
			os.system('scp {0} ashesh@172.24.68.124:/home/ashesh/project/NN/RNNexp/crf_to_rnn/CRFProblems/H3.6m/dataParser/Utils/{2}/{1}.dat'.format(fpath,fname,checkpoint_dir))
		#fname = 'train_example_N_{0}'.format(n)
		#convertAndSave(fname)
		
		for e in range(epoches):
			#fname = 'train_error_epoch_{1}_N_{0}'.format(n,e)
			#convertAndSave(fname)
			fname = 'forecast_epoch_{0}_N_{1}'.format(e,n)
			convertAndSave(fname)	
			fpath = path_to_trajfiles + fname
			if os.path.exists(fpath):
				os.system('scp {0} ashesh@172.24.68.124:/home/ashesh/project/NN/RNNexp/crf_to_rnn/CRFProblems/H3.6m/dataParser/Utils/{2}/{1}.dat'.format(fpath,fname,checkpoint_dir))
		for iterations in range(2000,6250,250):
			#fname = 'train_error_epoch_{1}_N_{0}'.format(n,e)
			#convertAndSave(fname)
			fname = 'forecast_iteration_{0}_N_{1}'.format(iterations,n)
			convertAndSave(fname)	
			fpath = path_to_trajfiles + fname
			if os.path.exists(fpath):
				os.system('scp {0} ashesh@172.24.68.124:/home/ashesh/project/NN/RNNexp/crf_to_rnn/CRFProblems/H3.6m/dataParser/Utils/{2}/{1}.dat'.format(fpath,fname,checkpoint_dir))
		
