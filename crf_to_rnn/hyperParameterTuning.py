import subprocess as sbp
import os
import copy
import socket as soc
from datetime import datetime

base_dir = ''
gpus = []
if soc.gethostname() == "napoli110.stanford.edu":
	#base_dir = '/scr/ashesh/h3.6m'
	base_dir = '/scail/scratch/group/cvgl/ashesh/h3.6m'
	gpus = [2]
elif soc.gethostname() == "napoli105.stanford.edu":
	#base_dir = '/scr/ashesh/h3.6m'
	base_dir = '/scail/scratch/group/cvgl/ashesh/h3.6m'
	gpus = [0]
elif soc.gethostname() == "napoli106.stanford.edu":
	#base_dir = '/scr/ashesh/h3.6m'
	base_dir = '/scail/scratch/group/cvgl/ashesh/h3.6m'
	gpus = [0]
elif soc.gethostname() == "napoli107.stanford.edu":
	#base_dir = '/scr/ashesh/h3.6m'
	base_dir = '/scail/scratch/group/cvgl/ashesh/h3.6m'
	gpus = [0]
elif soc.gethostname() == "napoli108.stanford.edu":
	#base_dir = '/scr/ashesh/h3.6m'
	base_dir = '/scail/scratch/group/cvgl/ashesh/h3.6m'
	gpus = [1]
elif soc.gethostname() == "napoli101.stanford.edu":
	#base_dir = '/scr/ashesh/h3.6m'
	base_dir = '/scail/scratch/group/cvgl/ashesh/h3.6m'
	gpus = [1]
elif soc.gethostname() == "napoli104.stanford.edu":
	#base_dir = '/scr/ashesh/h3.6m'
	base_dir = '/scail/scratch/group/cvgl/ashesh/h3.6m'
	gpus = [1]
elif soc.gethostname() == "napoli109.stanford.edu":
	#base_dir = '/scr/ashesh/h3.6m'
	base_dir = '/scail/scratch/group/cvgl/ashesh/h3.6m'
	gpus = [2]
elif soc.gethostname() == "ashesh":
	base_dir = '.'
else:
	print 'Incorrect machine'

cv = {}
cv_over = ['initial_lr','clipnorm','noise','decay']
#cv_over = ['initial_lr','clipnorm','maxiter']
cv['initial_lr'] = [1e-3]
cv['clipnorm'] = [25.0]
#cv['train_for'] = ['smoking','eating','discussion']
#cv['maxiter'] = [1000]
cv['noise'] = [	#[[],[]]
		#[[900.0,2000.0,3000.0,4000.0,5000.0,6000.0,7000.0,8000.0],[0.1,0.3,0.5,0.7,0.8,0.9,1.0,1.2]]
		[[250,0.5e3,1e3,1.3e3,2e3,2.5e3,3.3e3],[0.01,0.05,0.1,0.2,0.3,0.5,0.7]]
		#[[250,0.5e3,1e3,1.3e3,2e3,2.5e3,3.3e3],[0.01,0.05,0.1,0.2,0.3,0.5,0.65]]
		#[[1e3,1.5e3,1.8e3,2.3e3,3e3,3.8e3,4e3,4.5e3],[0.1,0.3,0.5,0.7,0.8,0.9,1.0,1.2]]
		#[[250,500,750,1000,1250,1500,1750,2000,2250,2500],[0.001,0.002,0.004,0.008,0.016,0.032,0.064,0.128,0.256,0.512]]	
		#[[250,500,750,1000,1250,1500,1750,2000,2250,2500,3300,4000],[0.001,0.002,0.004,0.008,0.016,0.032,0.064,0.128,0.256,0.512,0.65,0.7]]	
		]
		 
cv['decay'] = [	#[[1e3,4e3],[0.1,0.1]]
		[[1.5e3,4.5e3],[0.1,0.1]]	
		#[[4e3,7e3],[0.1,0.1]]
		]
#cv['noise_schedule'] = [[0.5e3,1e3,1.3e3,2e3,2.5e3,3.3e3,4e3,4.5e3]] #[[0.9e3,2e3,3e3,4e3,5e3,6e3,7e3,8e3]]
#cv['noise_rate_schedule'] =[[0.1,0.3,0.5,0.7,0.8,0.9,1.0,1.2]] #[[0.1,0.3,0.5,0.7,0.8,0.9,1.0,1.2]]
#cv['decay_schedule'] = [[1.5e3,4.5e3]]
#cv['decay_rate_schedule'] = [[0.1,0.1]] 
#noise [0.5e3,1e3,1.3e3,2e3,2.5e3,3.3e3,4e3,4.5e3],[0.1,0.3,0.5,0.7,0.8,0.9,1.0,1.2]],
#		 [[1e3,1.5e3,1.8e3,2.3e3,3e3,3.8e3,4e3,4.5e3],[0.1,0.3,0.5,0.7,0.8,0.9,1.0,1.2]]
def cvParams(cv_over,cv):
	new_list = []
	if len(cv_over) == 1:
		for x in cv[cv_over[0]]:
			new_list.append([x])
	else:
		permute_list = cvParams(cv_over[1:],cv)
		for x in cv[cv_over[0]]:
			for l in permute_list:
				nl = copy.deepcopy(l)
				nl.insert(0,x)
				new_list.append(nl)
	return new_list

cv_list = cvParams(cv_over,cv)
print cv_list

params = {}
params['decay_type'] = 'schedule'
params['decay_after'] =  -1
params['initial_lr'] = 1e-3
params['learning_rate_decay'] = 1.0
params['decay_schedule'] = []
params['decay_rate_schedule'] = [] 
params['lstm_init'] = 'uniform'
params['fc_init'] = 'uniform'
params['epochs'] = 2000
params['clipnorm'] = 0.0
params['use_noise'] = 1
params['noise_schedule'] = []
params['noise_rate_schedule'] = []
params['momentum'] = 0.99
params['g_clip'] = 25.0

params['truncate_gradient'] = 100
params['use_pretrained'] = 0
params['iter_to_load'] = 2500
params['model_to_train'] = 'dra'
params['sequence_length'] = 150
params['sequence_overlap'] = 50
params['batch_size'] = 100
params['lstm_size'] = 512
params['node_lstm_size'] = 512
params['fc_size'] = 256
params['snapshot_rate'] = 250
params['crf'] = ''
params['copy_state'] = 0
params['full_skeleton'] = 1
params['weight_decay'] = 0.0
params['train_for'] = 'eating'
params['temporal_features'] = 0
params['dra_type'] = 'NoEdge'
params['dataset_prefix'] = ''
params['drop_features'] = 0
params['drop_id'] = '9'
params['subsample_data'] = 1

'''
#Malik
params['truncate_gradient'] = 100
params['use_pretrained'] = 1
params['iter_to_load'] = 1250
params['model_to_train'] = 'lstm'
params['sequence_length'] = 150
params['sequence_overlap'] = 50
params['batch_size'] = 100
params['lstm_size'] = 1000
params['node_lstm_size'] = 1000
params['fc_size'] = 500
params['snapshot_rate'] = 250
params['crf'] = ''
params['copy_state'] = 0
params['full_skeleton'] = 1
params['weight_decay'] = 0.0
params['train_for'] = 'eating'
params['temporal_features'] = 0
params['dra_type'] = 'simple'
params['dataset_prefix'] = ''
params['drop_features'] = 0
params['drop_id'] = '9'

'''


load_pretrained_model_from = '/scail/scratch/group/cvgl/ashesh/h3.6m/checkpoints_lstm_T_150_bs_100_tg_100_initial_lr_0.001_clipnorm_25.0_nschd_[250,500.0,1000.0,1300.0,2000.0,2500.0,3300.0]_nrate_[0.01,0.05,0.1,0.2,0.3,0.5,0.7]_decschd_[1500.0,4500.0]_decrate_[0.1,0.1]_fs_discussion'

ongoing_file = open('ongoing_exp','a')
ongoing_file.write('=====\n'+str(datetime.now())+'\n')
ongoing_file.write(soc.gethostname()+'\n')

def listToString(ll):
	st = '['
	for v_ in ll:
		st += str(v_) + ','
	st = st[:-1] + ']'
	return st 


my_env = os.environ

# Adding CUDA to path
my_env['PATH'] += ':/usr/local/cuda/bin'

use_gpu = 0
for value_list in cv_list:
	if len(gpus) > 0:
		if use_gpu >= len(gpus):
			use_gpu = 0
		my_env['THEANO_FLAGS']='mode=FAST_RUN,device=gpu{0},floatX=float32'.format(gpus[use_gpu])
		use_gpu += 1
	else:
		my_env['THEANO_FLAGS']='mode=FAST_RUN,device=cpu,floatX=float32'.format(use_gpu)
	ongoing_file.write(my_env['THEANO_FLAGS']+'\n')
	if params['model_to_train'] == 'dra':
		params['checkpoint_path'] = 'checkpoints_{0}_T_{2}_bs_{1}_tg_{3}_ls_{4}_fc_{5}_'.format(params['model_to_train'],params['batch_size'],params['sequence_length'],params['truncate_gradient'],params['lstm_size'],params['fc_size'])
		if not params['node_lstm_size'] == params['lstm_size']:
			params['checkpoint_path'] += 'nls_{0}_'.format(params['node_lstm_size'])
	
	else:
		params['checkpoint_path'] = 'checkpoints_{0}_T_{2}_bs_{1}_tg_{3}_'.format(params['model_to_train'],params['batch_size'],params['sequence_length'],params['truncate_gradient'])
	for v,k  in zip(value_list,cv_over):
		
		st = ''
		if isinstance(v,list):
			if k == 'noise':
				params['noise_schedule'] = v[0]
				params['noise_rate_schedule'] = v[1]
				params['checkpoint_path'] +=  'nschd_{0}_nrate_{1}_'.format(listToString(v[0]),listToString(v[1]))

			if k == 'decay':
				params['decay_schedule'] = v[0]
				params['decay_rate_schedule'] = v[1]
				params['checkpoint_path'] +=  'decschd_{0}_decrate_{1}_'.format(listToString(v[0]),listToString(v[1]))
		else:
			params[k] = v
			st = v
			params['checkpoint_path'] +=  '{0}_{1}_'.format(k,st)
		
	params['checkpoint_path'] = params['checkpoint_path'][:-1] + params['crf']


	if params['weight_decay'] > 1e-6:
		params['checkpoint_path'] += '_wd_{0}'.format(params['weight_decay'])
	
	if params['full_skeleton']:
		params['checkpoint_path'] += '_fs'

	if not params['train_for'] == 'validate':
		params['checkpoint_path'] += '_' + params['train_for']

	if params['temporal_features']:
		params['checkpoint_path'] += '_tf'

	if params['drop_features']:
		params['checkpoint_path'] += '_df_' + params['drop_id']

	if (not params['dra_type'] == 'simple') and params['model_to_train'] == 'dra':
		params['checkpoint_path'] += '_' + params['dra_type']

	if len(params['dataset_prefix']) > 0:
		params['checkpoint_path'] += params['dataset_prefix']
	
	if not params['subsample_data']:
		params['checkpoint_path'] += '_fullrate'


	path_to_checkpoint = base_dir + '/{0}/'.format(params['checkpoint_path'])
	if not os.path.exists(path_to_checkpoint):
		os.mkdir(path_to_checkpoint)
	
	if params['use_pretrained'] == 1:
		if load_pretrained_model_from[-1] == '/':
			os.system('cp {0}checkpoint.{1} {2}.'.format(load_pretrained_model_from,params['iter_to_load'],path_to_checkpoint))
			os.system('cp {0}logfile {1}.'.format(load_pretrained_model_from,path_to_checkpoint))
			os.system('cp {0}complete_log {1}.'.format(load_pretrained_model_from,path_to_checkpoint))
		else:
			os.system('cp {0}/checkpoint.{1} {2}.'.format(load_pretrained_model_from,params['iter_to_load'],path_to_checkpoint))
			os.system('cp {0}/logfile {1}.'.format(load_pretrained_model_from,path_to_checkpoint))
			os.system('cp {0}/complete_log {1}.'.format(load_pretrained_model_from,path_to_checkpoint))

	print 'Dir: {0}'.format(path_to_checkpoint)
	args = ['python','trainDRA.py']
	for k in params.keys():
		args.append('--{0}'.format(k))
		if not isinstance(params[k],list):
			args.append(str(params[k]))
		else:
			for x in params[k]:
				args.append(str(x))

	FNULL = open('{0}stdout.txt'.format(path_to_checkpoint),'w')
	p=sbp.Popen(args,env=my_env,shell=False,stdout=FNULL,stderr=sbp.STDOUT)
	pd = p.pid
	#p.wait()
	ongoing_file.write('{0} {1}\n'.format(path_to_checkpoint,pd))
ongoing_file.close()
