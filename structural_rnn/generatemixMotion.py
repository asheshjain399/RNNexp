import subprocess as sbp
import os
import copy
import socket as soc
from datetime import datetime

my_env = os.environ
# Adding CUDA to path
my_env['PATH'] += ':/usr/local/cuda/bin'
my_env['THEANO_FLAGS']='mode=FAST_RUN,device=gpu0,floatX=float32'

params = {}
params['forecast'] = 'dra'
params['checkpoint'] = 'checkpoints_dra_T_150_bs_100_tg_100_ls_512_fc_256_initial_lr_0.001_clipnorm_25.0_nschd_[250,500.0,1000.0,1300.0,2000.0,2500.0,3300.0]_nrate_[0.01,0.05,0.1,0.2,0.3,0.5,0.7]_decschd_[1500.0,4500.0]_decrate_[0.1,0.1]_dl_fs_final'
#params['checkpoint2'] = 'checkpoints_dra_T_150_bs_100_tg_100_ls_512_fc_256_initial_lr_0.001_clipnorm_25.0_nschd_[250,500.0,1000.0,1300.0,2000.0,2500.0,3300.0]_nrate_[0.01,0.05,0.1,0.2,0.3,0.5,0.7]_decschd_[1500.0,4500.0]_decrate_[0.1,0.1]_fs_final'
params['checkpoint2'] = 'checkpoints_dra_T_150_bs_100_tg_100_ls_512_fc_256_initial_lr_0.001_clipnorm_25.0_nschd_[250,500.0,1000.0,1300.0,2000.0,2500.0,3300.0]_nrate_[0.01,0.05,0.1,0.2,0.3,0.5,0.7]_decschd_[1500.0,4500.0]_decrate_[0.1,0.1]_fs_final_fullrate'
params['swap_edgernn'] = ['legleft_input']
params['swap_nodernn'] = ['legleft']
params['swap_edgernn2'] = ['leg_input']
params['swap_nodernn2'] = ['leg']
params['iteration'] = 4000
params['iteration2'] = 4000
params['motion_prefix'] = 50
params['motion_suffix'] = 100
params['subsample_data'] = 1
params['crf'] = '_dl'
args = ['python','mixskeleton.py']
for k in params.keys():
	args.append('--{0}'.format(k))
	if not isinstance(params[k],list):
		args.append(str(params[k]))
	else:
		for x in params[k]:
			args.append(str(x))
p=sbp.Popen(args,env=my_env)#,shell=False,stderr=sbp.STDOUT)
p.wait()
