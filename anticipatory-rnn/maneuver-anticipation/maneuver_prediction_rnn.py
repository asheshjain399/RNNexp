import numpy as np
import theano
import os
from theano import tensor as T
from neuralmodels.utils import permute, load
from neuralmodels.costs import softmax_decay_loss,softmax_loss
from neuralmodels.models import RNN
from neuralmodels.predictions import OutputMaxProb, OutputSampleFromDiscrete,OutputActionThresh
from neuralmodels.layers import softmax, simpleRNN, OneHot, LSTM, TemporalInputFeatures
import cPickle
from utils import confusionMat
from predictions import predictManeuver,predictLastTimeManeuver
import sys

def evaluate(index,fold,checkpoint):
	path_to_dataset = '/scr/ashesh/brain4cars/dataset/{0}'.format(fold)
	path_to_checkpoints = '/scr/ashesh/brain4cars/checkpoints/{0}'.format(fold)
	test_data = cPickle.load(open('{1}/test_data_{0}.pik'.format(index,path_to_dataset)))	
	Y_te = test_data['labels']
	X_te = test_data['features']
	actions = []
	if test_data.has_key('actions'):
		actions = test_data['actions']
	else:
		actions = ['end_action','lchange','rchange','lturn','rturn']

	# Prediction
	rnn = load('{2}/{0}/checkpoint.{1}'.format(index,checkpoint,path_to_checkpoints))

	predictions = []
	errors = 0
	N = 0
	P = []
	Y = []
	for xte,yte in zip(X_te,Y_te):
		prediction = rnn.predict_output(xte,OutputActionThresh)
		#print prediction.T
		predictions.append(prediction)
		t = np.nonzero(yte-prediction)
	
		# Label 0 is the dummy label. Currently maneuvers are labeled [1..n]	
		prediction = prediction[:,0]
		actual = yte[:,0]
		prediction[prediction>0] -= 1
		actual[actual>0] -= 1
		
		p = predictManeuver(prediction,actions)
		y = actual[-1]
		P.append(p)
		Y.append(y)
		result = {'actual':y,'prediction':p,'timeseries':list(prediction)}
		#print result.values()
		errors += len(t[0])
		N += yte.shape[0]
	
	P = np.array(P)
	Y = np.array(Y)
	[conMat,p_mat,re_mat] = confusionMat(P,Y)

	cPickle.dump(predictions,open('{1}/prediction_{0}.pik'.format(index,path_to_dataset),'wb'))
	return conMat,p_mat,re_mat

def load_rnn(index,fold,checkpoint):
	path_to_dataset = '/scr/ashesh/brain4cars/dataset/{0}'.format(fold)
	path_to_checkpoints = '/scr/ashesh/brain4cars/checkpoints/{0}'.format(fold)
	rnn = load('{2}/{0}/checkpoint.{1}'.format(index,checkpoint,path_to_checkpoints))
	return rnn

if __name__ == '__main__':
	index = sys.argv[1]	
	fold = sys.argv[2]	
	checkpoint = sys.argv[3]
	[conMat,p_mat,re_mat] = evaluate(index,fold,checkpoint)
	print conMat
	print p_mat
	print re_mat

