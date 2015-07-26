import numpy as np
from maneuver_prediction_rnn import evaluate 

folds = ['fold_1', 'fold_2', 'fold_3', 'fold_4', 'fold_5']

#index = ['054061', '152011', '646142', '217335', '775723']
#extra_samples=2

#index = ['159003', '252991', '738032', '460969', '103934']
#extra_samples=4
# exp(-t) th=0.68, epoch=300. Pr=77.2 Re=74.3 Adagrad() two layers

index = ['960453', '298544', '558591', '463107', '362130']
#extra_samples=4
# exp(-t) th=0.68, epoch=300 Pr=78.4 Re=71.7 Adagrad() only one layer 

size=5
confMat = np.zeros((size,size))
pMat = np.zeros((size,size))
rMat = np.zeros((size,size))
avg_precision = 0.0
avg_recall = 0.0
for i,f in zip(index,folds):
	[conMat,p_mat,re_mat] = evaluate(i,f,'300')
	confMat += conMat
	pMat += p_mat
	rMat += re_mat
	avg_precision += np.mean(np.diag(p_mat)[1:])
	avg_recall += np.mean(np.diag(re_mat)[1:])

avg_precision *= 0.2
avg_recall *= 0.2

print "Precision"
print 0.2*pMat
print avg_precision

print "Recall"
print 0.2*rMat
print avg_recall	
