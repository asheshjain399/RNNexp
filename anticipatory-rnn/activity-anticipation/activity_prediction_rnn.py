import numpy as np
import os
import sys
import cPickle
from theano import tensor as T
from readData import sortActivities
from neuralmodels.utils import permute, load
from neuralmodels.costs import softmax_loss
from neuralmodels.models import RNN
from neuralmodels.predictions import OutputMaxProb, OutputSampleFromDiscrete
from neuralmodels.layers import softmax, simpleRNN, OneHot, LSTM, TemporalInputFeatures

def predict_activitiy(index,fold,checkpoint):
	path_to_dataset = '/scr/ashesh/activity-anticipation/dataset/{0}'.format(fold)
	path_to_checkpoints = '/scr/ashesh/activity-anticipation/checkpoints/{0}'.format(fold)

	test_data = cPickle.load(open('{1}/test_data_{0}.pik'.format(index,path_to_dataset)))	
	Y_te = test_data['labels']
	X_te = test_data['features']

	ground_truth_test_data = cPickle.load(open('{1}/ground_truth_test_data_{0}.pik'.format(index,path_to_dataset)))	
	Y_te_ground_truth = ground_truth_test_data['labels']

	rnn = load('{2}/{0}/checkpoint.{1}'.format(index,checkpoint,path_to_checkpoints))

	predictions = []
	errors = 0
	N = 0
	for xte,yte,gte in zip(X_te,Y_te,Y_te_ground_truth):
		prediction = rnn.predict_output(xte,OutputMaxProb)
		predictions.append(prediction)
		t = np.nonzero(yte-prediction)
		
		'''
		print "{0}".format(gte[:,0])
		print "{0}".format(yte[:,0])
		print "{0}".format(prediction[:,0])
		print ""
		#print t
		'''

		errors += len(t[0])
		N += yte.shape[0]
	#cPickle.dump(predictions,open('{1}/prediction_{0}.pik'.format(index,path_to_dataset),'wb'))
	print 'error = {0}'.format(errors*1.0/N)
	return (errors*1.0/N)

if __name__ == "__main__":
	index = sys.argv[1]	
	#fold = sys.argv[2]
	checkpoint = sys.argv[2]
	folds = ['1','2','3','4']
	err = []	
	for fold in folds:
		err.append(predict_activitiy(index,'fold_'+fold,checkpoint))
	err = np.array(err)
	print err
	print '{0} ({1})'.format(np.mean(err),np.std(err))
