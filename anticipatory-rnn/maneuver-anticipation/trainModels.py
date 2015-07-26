import numpy as np
import os

folds = ['fold_1', 'fold_2', 'fold_3', 'fold_4', 'fold_5']

#index = ['054061', '152011', '646142', '217335', '775723']
#extra_samples=2

#index = ['159003', '252991', '738032', '460969', '103934']
#extra_samples=4
# exp(-t) th=0.68, epoch=300. Pr=77.2 Re=74.3 Adagrad()

index = ['960453', '298544', '558591', '463107', '362130']
#extra_samples=4
# exp(-t*t) th=0.68, 

for i,f in zip(index,folds):
	os.system('THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python maneuver-rnn.py {0} {1}'.format(i,f))
