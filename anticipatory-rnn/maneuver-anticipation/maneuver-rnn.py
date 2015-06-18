import sys
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

if __name__ == '__main__':

	index = sys.argv[1]	
	path_to_dataset = '/scr/ashesh/brain4cars/dataset'
	path_to_checkpoints = '/scr/ashesh/brain4cars/checkpoints'
	test_data = cPickle.load(open('{1}/test_data_{0}.pik'.format(index,path_to_dataset)))	
	Y_te = test_data['labels']
	X_te = test_data['features']
	actions = []
	if test_data.has_key('actions'):
		actions = test_data['actions']
	else:
		actions = ['end_action','lchange','rchange','lturn','rturn']
	#print X_te.shape
	#print Y_te.shape

	train_data = cPickle.load(open('{1}/train_data_{0}.pik'.format(index,path_to_dataset)))	
	Y_tr = train_data['labels']
	X_tr = train_data['features']
	print X_tr.shape
	print Y_tr.shape

	
	print type(X_tr[0,0,0])

	num_train = X_tr.shape[1]
	num_test = len(X_te)
	len_samples = X_tr.shape[0]

	num_classes = int(np.max(Y_tr) - np.min(Y_tr) + 1)
	inputD = X_tr.shape[2]
	outputD = num_classes 

	print 'Number of classes ',outputD
	print 'Feature dimension ',inputD

	epochs = 2000
	batch_size = num_train
	learning_rate_decay = 0.97
	decay_after = 5
	
	use_pretrained = False 
	train_more = False
	global rnn
	if not use_pretrained:
		# Creating network layers
		layers = [TemporalInputFeatures(inputD),LSTM('tanh','sigmoid','orthogonal',6,16,None),softmax(num_classes)]

		trY = T.lmatrix()
		

		# Initializing network
		rnn = RNN(layers,softmax_decay_loss,trY,1e-3)

		if not os.path.exists(path_to_checkpoints):
			os.mkdir(path_to_checkpoints)

		if not os.path.exists('{1}/{0}/'.format(index,path_to_checkpoints)):
			os.mkdir('{1}/{0}/'.format(index,path_to_checkpoints))

		# Fitting model
		rnn.fitModel(X_tr,Y_tr,1,'{1}/{0}/'.format(index,path_to_checkpoints),epochs,batch_size,learning_rate_decay,decay_after)
	else:
		checkpoint = sys.argv[2]
		# Prediction
		rnn = load('{2}/{0}/checkpoint.{1}'.format(index,checkpoint,path_to_checkpoints))
		if train_more:
			rnn.fitModel(X_tr,Y_tr,1,'{1}/{0}/'.format(index,path_to_checkpoints),epochs,batch_size,learning_rate_decay,decay_after)


	predictions = []
	errors = 0
	N = 0
	P = []
	Y = []
	for xte,yte in zip(X_te,Y_te):
		prediction = rnn.predict_output(xte,OutputActionThresh)
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

		errors += len(t[0])
		N += yte.shape[0]
	
	P = np.array(P)
	Y = np.array(Y)
	[conMat,p_mat,re_mat] = confusionMat(P,Y)
	print conMat
	print p_mat
	print re_mat

	cPickle.dump(predictions,open('{1}/prediction_{0}.pik'.format(index,path_to_dataset),'wb'))
	print 'error = {0}'.format(errors*1.0/N)
	#cPickle.dump(Y_te,open('test.pik','wb'))

	
