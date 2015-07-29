import numpy as np
from maneuver_prediction_rnn import evaluate 
import sys

maneuver_type = sys.argv[1]
checkpoint = sys.argv[2]

folds = ['fold_1', 'fold_2', 'fold_3', 'fold_4', 'fold_5']

size = 5
index = []
if maneuver_type == 'all':
	size = 5
	#index = ['054061', '152011', '646142', '217335', '775723']
	#extra_samples=2

	#index = ['159003', '252991', '738032', '460969', '103934']
	#extra_samples=4
	# exp(-t) th=0.68, epoch=300. Pr=77.2 Re=74.3 Adagrad() two layers

	index = ['960453', '298544', '558591', '463107', '362130']
	#extra_samples=4
	# exp(-t) th=0.68, epoch=300 Pr=78.4 Re=71.7 Adagrad() only one layer 
elif maneuver_type == 'lane':
	index = ['723759', '723759', '723759', '723759', '723759']
	size = 3		
elif maneuver_type == 'turns':
	index = ['723759', '723759', '723759', '723759', '723759']
	size = 3
else:
	print "did not match any maneuver"

confMat = np.zeros((size,size))
pMat = np.zeros((size,size))
rMat = np.zeros((size,size))

avg_precision = []
avg_recall = []
avg_anticipation_time = []

for i,f in zip(index,folds):
	[conMat,p_mat,re_mat,time_mat] = evaluate(i,f,checkpoint)
	confMat += conMat
	pMat += p_mat
	rMat += re_mat
	avg_precision.append(np.mean(np.diag(p_mat)[1:]))
	avg_recall.append(np.mean(np.diag(re_mat)[1:]))
	avg_anticipation_time.append(np.mean(  np.divide(np.diag(time_mat)[1:],np.diag(conMat)[1:])   ))

avg_precision = np.array(avg_precision)
avg_recall = np.array(avg_recall)
avg_anticipation_time = np.array(avg_anticipation_time)


print "Precision"
print 0.2*pMat
print "Precision = {0} ({1})".format(np.mean(avg_precision),(np.std(avg_precision)/np.sqrt(len(folds))))

print "Recall"
print 0.2*rMat
print "Recall = {0} ({1})".format(np.mean(avg_recall),(np.std(avg_recall)/np.sqrt(len(folds))))

print "Anticipation time"
print "Time = {0} ({1})".format(np.mean(avg_anticipation_time),(np.std(avg_anticipation_time)/np.sqrt(len(folds))))
