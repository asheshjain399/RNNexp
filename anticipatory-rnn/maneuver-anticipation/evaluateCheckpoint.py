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
import copy

def evaluate(path_to_dataset,path_to_checkpoint,model_type='multipleRNNs'):
	'''
	Input: 
	path_to_dataset: Complete path to the pickle file containing data
	path_to_checkpoint: Complete path to the checkpoint to evaluate at
	model_type: the kind of model to load. This is the same model using which the checkpoint was created

	Before running this function make sure that the threshold at which you want to evaluate is written in settings.py file. 
	'''

	test_data = cPickle.load(open(path_to_dataset))	
	Y_te = test_data['labels']
	X_te = test_data['features']
	actions = []
	if test_data.has_key('actions'):
		actions = test_data['actions']
	else:
		actions = ['end_action','lchange','rchange','lturn','rturn']

	rnn = []
	if model_type == 'multipleRNNs':
			rnn = loadMultipleRNNsCombined(path_to_checkpoint)
	else:
			rnn = load(path_to_checkpoint)

	print "Number of parameters {0}".format(rnn.num_params)
	
	conMat = {}
	p_mat = {}
	re_mat = {}
	time_mat = {}

	P = {}
	Y = {}
	for i in range(10):
		P[i] = []
		Y[i] = []
		conMat[i] = [] 
		p_mat[i] = []
		re_mat[i] = []
		time_mat[i] = []
	i = 'best'
	P[i] = []
	Y[i] = []
	conMat[i] = [] 
	p_mat[i] = []
	re_mat[i] = []
	time_mat[i] = []

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

	
		# Label 0 is the dummy label. Currently maneuvers are labeled [1..n]	
		prediction = prediction[:,0]
		actual = yte[:,0]
		prediction[prediction>0] -= 1
		actual[actual>0] -= 1	
		y = actual[-1]
	
		iter_ = 10 - len(prediction) 
		for p in prediction:
			P[iter_].append(p)
			Y[iter_].append(y)
			iter_ += 1

		p,anticipation_time = predictManeuver(prediction,actions)
		P['best'].append(p)
		Y['best'].append(y)
		Time_before_maneuver.append(anticipation_time)

	Time_before_maneuver = np.array(Time_before_maneuver)
	for k in P.keys():	
		if len(P[k]) == 0:
			continue
		P[k] = np.array(P[k])
		Y[k] = np.array(Y[k])
		[conMat_,p_mat_,re_mat_,time_mat_] = confusionMat(P[k],Y[k],Time_before_maneuver)

		conMat[k] = conMat_
		p_mat[k] = p_mat_
		re_mat[k] = re_mat_
		time_mat[k] = time_mat_

	return conMat,p_mat,re_mat,time_mat

'''
Input: 
path_to_dataset: Complete path to the pickle file containing data
path_to_checkpoint: Complete path to the checkpoint to evaluate at
model_type: the kind of model to load. This is the same model using which the checkpoint was created

Before running this function make sure that the threshold at which you want to evaluate is written in settings.py file. 
'''
'''
def evaluate(path_to_dataset,path_to_checkpoint,model_type='multipleRNNs'):

	test_data = cPickle.load(open(path_to_dataset))	
	Y_te = test_data['labels']
	X_te = test_data['features']
	actions = []
	if test_data.has_key('actions'):
		actions = test_data['actions']
	else:
		actions = ['end_action','lchange','rchange','lturn','rturn']

	rnn = []
	if model_type == 'multipleRNNs':
			rnn = loadMultipleRNNsCombined(path_to_checkpoint)
	else:
			rnn = load(path_to_checkpoint)

	print "Number of parameters {0}".format(rnn.num_params)

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
'''

def evaluateForAllThresholds(path_to_dataset,path_to_checkpoint,thresh_params,model_type='multipleRNNs'):
	'''
	Input: 
	path_to_dataset: Complete path to the pickle file containing data
	path_to_checkpoint: Complete path to the checkpoint to evaluate at
	thresh_params: The list of threshold values to  evaluate at
	model_type: the kind of model to load. This is the same model using which the checkpoint was created

	This function evaluates (data,checkpoint) for all the values of threshold specified in thresh_params
	'''
	test_data = cPickle.load(open(path_to_dataset))	
	Y_te = test_data['labels']
	X_te = test_data['features']
	actions = []
	if test_data.has_key('actions'):
		actions = test_data['actions']
	else:
		actions = ['end_action','lchange','rchange','lturn','rturn']

	rnn = []
	if model_type == 'multipleRNNs':
			rnn = loadMultipleRNNsCombined(path_to_checkpoint)
	else:
			rnn = load(path_to_checkpoint)

	precision = []
	recall = []
	time_before_maneuver = []

	for th in thresh_params:
		with open('settings.py','w') as f:
			f.write('OUTPUT_THRESH = %f \n' % th)
		print "Generating results for th= ",th
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
			yte_ = copy.deepcopy(yte)
			actual = yte_[:,0]
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
		precision.append(np.mean(np.diag(p_mat)[1:]))
		recall.append(np.mean(np.diag(re_mat)[1:]))
		time_before_maneuver.append(np.mean(  np.divide(np.diag(time_mat)[1:],np.diag(conMat)[1:])   ))

	return np.array(precision),np.array(recall),np.array(time_before_maneuver)

