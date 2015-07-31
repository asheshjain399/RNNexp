import numpy as np
import theano
import os
from theano import tensor as T
from neuralmodels.utils import permute, load, loadMultipleRNNsCombined
from neuralmodels.costs import softmax_decay_loss,softmax_loss
from neuralmodels.models import RNN, MultipleRNNsCombined
from neuralmodels.predictions import OutputMaxProb, OutputSampleFromDiscrete,OutputActionThresh
from neuralmodels.layers import softmax, simpleRNN, OneHot, LSTM, TemporalInputFeatures
import cPickle
from utils import confusionMat
from predictions import predictManeuver,predictLastTimeManeuver
import sys

def evaluate(index,fold,checkpoint,model_type='lstm_one_layer',path_to_load_from=''):
	path_to_dataset = '/scr/ashesh/brain4cars/dataset/{0}'.format(fold)

	if len(path_to_load_from) > 0:
		path_to_dataset = path_to_load_from

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
	rnn = []
	if model_type == 'multipleRNNs':
		if len(path_to_load_from) > 0:
			rnn = loadMultipleRNNsCombined('{0}/checkpoint.{1}'.format(path_to_load_from,checkpoint))
		else:
			rnn = loadMultipleRNNsCombined('{2}/{0}/checkpoint.{1}'.format(index,checkpoint,path_to_checkpoints))
	else:
		if len(path_to_load_from) > 0:
			rnn = load('{0}/checkpoint.{1}'.format(path_to_load_from,checkpoint))
		else:
			rnn = load('{2}/{0}/checkpoint.{1}'.format(index,checkpoint,path_to_checkpoints))

	predictions = []
	errors = 0
	N = 0
	P = []
	Y = []
	Time_before_maneuver = []
	for xte,yte in zip(X_te,Y_te):
		inputD = xte.shape[2]
		road_feature_dimension = 4
		prediction = []

		if model_type == 'multipleRNNs':
			prediction = rnn.predict_output([xte[:,:,(inputD-road_feature_dimension):],xte[:,:,:(inputD-road_feature_dimension)]],OutputActionThresh)
			#[:,:,:(inputD-road_feature_dimension)]
		else:
			prediction = rnn.predict_output(xte,OutputActionThresh)

		#print prediction.T
		predictions.append(prediction)
		t = np.nonzero(yte-prediction)
	
		# Label 0 is the dummy label. Currently maneuvers are labeled [1..n]	
		prediction = prediction[:,0]
		actual = yte[:,0]
		prediction[prediction>0] -= 1
		actual[actual>0] -= 1
		
		p,anticipation_time = predictManeuver(prediction,actions)
		y = actual[-1]
		P.append(p)
		Y.append(y)
		Time_before_maneuver.append(anticipation_time)
		result = {'actual':y,'prediction':p,'timeseries':list(prediction)}
		#print result.values()
		errors += len(t[0])
		N += yte.shape[0]
	
	P = np.array(P)
	Y = np.array(Y)
	Time_before_maneuver = np.array(Time_before_maneuver)
	[conMat,p_mat,re_mat,time_mat] = confusionMat(P,Y,Time_before_maneuver)
	return conMat,p_mat,re_mat,time_mat

def load_rnn(index,fold,checkpoint):
	path_to_dataset = '/scr/ashesh/brain4cars/dataset/{0}'.format(fold)
	path_to_checkpoints = '/scr/ashesh/brain4cars/checkpoints/{0}'.format(fold)
	rnn = load('{2}/{0}/checkpoint.{1}'.format(index,checkpoint,path_to_checkpoints))
	return rnn

if __name__ == '__main__':
	index = sys.argv[1]	
	fold = sys.argv[2]	
	checkpoint = sys.argv[3]
	[conMat,p_mat,re_mat,time_mat] = evaluate(index,fold,checkpoint)
	print conMat
	print p_mat
	print re_mat

