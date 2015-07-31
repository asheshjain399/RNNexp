import numpy as np
from maneuver_prediction_rnn import evaluate 
import sys

maneuver_type = sys.argv[1]
checkpoint = sys.argv[2]

checkpoints = [checkpoint,checkpoint,checkpoint,checkpoint,checkpoint]

folds = ['fold_1', 'fold_2', 'fold_3', 'fold_4', 'fold_5']

size = 5
index = []
model_type = 'lstm_one_layer'
path_to_load_from = ''
append_to_checkpoint = ''
if maneuver_type == 'all':
	size = 5
	model_type = 'multipleRNNs'
	i = '356988'
	index = [i,i,i,i,i]
	checkpoints = ['350','420','420','350','400']
	path_to_load_from = 'checkpoints/all'
	append_to_checkpoint = '_multipleRNNs_lstm_size_64'
	# Set threshold in neuralmodels/prediction.py = 0.76

elif maneuver_type == 'lane':
	size = 3	
	i = '723759'	
	model_type = 'multipleRNNs'
	index = [i,i,i,i,i]
	checkpoints = ['200','420','200','250','200']
	path_to_load_from = 'checkpoints/lane'
	append_to_checkpoint = '_multipleRNNs_lstm_size_64'
	# Set threshold in neuralmodels/prediction.py = 0.8

elif maneuver_type == 'turns':
	size = 3
	model_type = 'multipleRNNs'
	i = '209221'
	index = [i,i,i,i,i]
	checkpoints = ['400','400','500','400','400']
	path_to_load_from = 'checkpoints/turns'
	append_to_checkpoint = '_multipleRNNs_lstm_size_64'
	# Set threshold in neuralmodels/prediction.py = 0.868
else:
	print "did not match any maneuver"

confMat = np.zeros((size,size))
pMat = np.zeros((size,size))
rMat = np.zeros((size,size))

avg_precision = []
avg_recall = []
avg_anticipation_time = []

for i,f,checkpoint in zip(index,folds,checkpoints):
	path_new = ''
	if len(path_to_load_from) > 0:
		checkpoint += append_to_checkpoint
		path_new = path_to_load_from + '/' + f
	[conMat,p_mat,re_mat,time_mat] = evaluate(i,f,checkpoint,model_type,path_new)
	confMat += conMat
	pMat += p_mat
	rMat += re_mat
	avg_precision.append(np.mean(np.diag(p_mat)[1:]))
	avg_recall.append(np.mean(np.diag(re_mat)[1:]))
	avg_anticipation_time.append(np.mean(  np.divide(np.diag(time_mat)[1:],np.diag(conMat)[1:])   ))

avg_precision = np.array(avg_precision)
avg_recall = np.array(avg_recall)
avg_anticipation_time = np.array(avg_anticipation_time)

'''
print "Precision"
print 0.2*pMat
print "Precision = {0} ({1})".format(np.mean(avg_precision),(np.std(avg_precision)/np.sqrt(len(folds))))

print "Recall"
print 0.2*rMat
print "Recall = {0} ({1})".format(np.mean(avg_recall),(np.std(avg_recall)/np.sqrt(len(folds))))

print "Anticipation time"
print "Time = {0} ({1})".format(np.mean(avg_anticipation_time),(np.std(avg_anticipation_time)/np.sqrt(len(folds))))
'''
print '****************{0}****************'.format(checkpoint)
print 'Precision'
print avg_precision
print "{0} ({1})".format(np.mean(avg_precision),(np.std(avg_precision)/np.sqrt(len(folds))))
print ''

print 'Recall'
print avg_recall
print "{0} ({1})".format(np.mean(avg_recall),(np.std(avg_recall)/np.sqrt(len(folds))))
print ''
