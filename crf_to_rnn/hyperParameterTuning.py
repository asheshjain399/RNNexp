import subprocess as sbp
import os
import copy
import socket as soc
from datetime import datetime

base_dir = ''
gpus = 0
if soc.gethostname() == "napoli110.stanford.edu":
	#base_dir = '/scr/ashesh/h3.6m'
	base_dir = '/scail/scratch/group/cvgl/ashesh/h3.6m'
	gpus = 1
if soc.gethostname() == "napoli106.stanford.edu":
	#base_dir = '/scr/ashesh/h3.6m'
	base_dir = '/scail/scratch/group/cvgl/ashesh/h3.6m'
	gpus = 2
if soc.gethostname() == "napoli107.stanford.edu":
	#base_dir = '/scr/ashesh/h3.6m'
	base_dir = '/scail/scratch/group/cvgl/ashesh/h3.6m'
	gpus = 2
elif soc.gethostname() == "ashesh":
	base_dir = '.'

cv = {}
cv_over = ['initial_lr','clipnorm','momentum','g_clip','noise_schedule','noise_rate_schedule']
cv['initial_lr'] = [1e-3]#,1e-4]
cv['clipnorm'] = [0.0,25.0]
cv['momentum'] = [0.99]#,0.99]
cv['g_clip'] = [25.0]
cv['noise_schedule'] = [[0.9e3,2e3,3e3,4e3,5e3,6e3,7e3,8e3]]
cv['noise_rate_schedule'] = [[0.1,0.3,0.5,0.7,0.8,0.9,1.0,1.2]]
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
params['decay_schedule'] = [1e3,4e3]
params['decay_rate_schedule'] = [0.1,0.1] 
params['lstm_size'] = 1000
params['lstm_init'] = 'uniform'
params['fc_init'] = 'uniform'
params['snapshot_rate'] = 20
params['epochs'] = 2000
params['clipnorm'] = 0.0
params['use_noise'] = 1
params['noise_schedule'] = [2,6,12,20,35,50,60,80]
params['noise_rate_schedule'] = [0.01,0.05,0.075,0.1,0.5,0.8,1.0,1.2]
params['momentum'] = 0.9
params['g_clip'] = 25.0
params['truncate_gradient'] = 100
params['use_pretrained'] = 0
params['model_to_train'] = 'lstm'

params['sequence_length'] = 500
params['batch_size'] = 20


ongoing_file = open('ongoing_exp','a')
ongoing_file.write('=====\n'+str(datetime.now())+'\n')
ongoing_file.write(soc.gethostname()+'\n')

my_env = os.environ
use_gpu = 0
for value_list in cv_list:
	if gpus > 0:
		if use_gpu >= gpus:
			use_gpu = 0
		my_env['THEANO_FLAGS']='mode=FAST_RUN,device=gpu{0},floatX=float32'.format(use_gpu)
		use_gpu += 1
	else:
		my_env['THEANO_FLAGS']='mode=FAST_RUN,device=cpu,floatX=float32'.format(use_gpu)
	ongoing_file.write(my_env['THEANO_FLAGS']+'\n')
	params['checkpoint_path'] = 'checkpoints_{0}_T_{2}_batch_size_{1}_'.format(params['model_to_train'],params['batch_size'],params['sequence_length'])
	for v,k  in zip(value_list,cv_over):
		params[k] = v
		st = ''
		if isinstance(v,list):
			st = '['
			for v_ in v:
				st += str(v_) + ','
			st = st[:-1] + ']'
		else:
			st = v
		params['checkpoint_path'] +=  '{0}_{1}_'.format(k,st)
	params['checkpoint_path'] = params['checkpoint_path'][:-1]

	path_to_checkpoint = base_dir + '/{0}/'.format(params['checkpoint_path'])
	if not os.path.exists(path_to_checkpoint):
		os.mkdir(path_to_checkpoint)
	
	print 'Dir: {0}'.format(path_to_checkpoint)
	FNULL = open('{0}stdout.txt'.format(path_to_checkpoint),'w')
	args = ['python','trainDRA.py']
	for k in params.keys():
		args.append('--{0}'.format(k))
		if not isinstance(params[k],list):
			args.append(str(params[k]))
		else:
			for x in params[k]:
				args.append(str(x))

	p=sbp.Popen(args,env=my_env)#,shell=False,stdout=FNULL,stderr=sbp.STDOUT)
	pd = p.pid
	p.wait()
	ongoing_file.write('{0} {1}\n'.format(path_to_checkpoint,pd))
ongoing_file.close()
