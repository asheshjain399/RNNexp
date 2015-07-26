import sys
import numpy as np
import theano
import os
from theano import tensor as T
from readData import sortActivities
from neuralmodels.utils import permute, load
from neuralmodels.costs import softmax_loss
from neuralmodels.models import RNN
from neuralmodels.predictions import OutputMaxProb, OutputSampleFromDiscrete
from neuralmodels.layers import softmax, simpleRNN, OneHot, LSTM, TemporalInputFeatures
import cPickle

def text_prediction(class_ids_reverse,p_labels):
	N = p_labels.shape[1]
	T = p_labels.shape[0]
	text_output = []
	for i in range(N):
		t = ''
		for j in p_labels[:,i]:
			t = t + class_ids_reverse[j]
		text_output.append(t)
	return text_output

def jointModel():
	shared_input_layer = TemporalInputFeatures(inputJointFeatures)
	shared_hidden_layer = LSTM('tanh','sigmoid','orthogonal',4,128)
	shared_layers = [shared_input_layer,shared_hidden_layer]
	human_layers = [ConcatenateFeatures(inputHumanFeatures),LSTM('tanh','sigmoid','orthogonal',4,256),softmax(num_sub_activities)]
	object_layers = [ConcatenateFeatures(inputObjectFeatures),LSTM('tanh','sigmoid','orthogonal',4,256),softmax(num_affordances)]



if __name__ == '__main__':

	index = sys.argv[1]	
	fold = sys.argv[2]	
	path_to_dataset = '/scr/ashesh/activity-anticipation/dataset/{0}'.format(fold)
	path_to_checkpoints = '/scr/ashesh/activity-anticipation/checkpoints/{0}'.format(fold)

	if not os.path.exists(path_to_checkpoints):
		os.mkdir(path_to_checkpoints)

	test_data = cPickle.load(open('{1}/test_data_{0}.pik'.format(index,path_to_dataset)))	
	Y_te = test_data['labels']
	X_te = test_data['features']

	#print X_te.shape
	#print Y_te.shape

	train_data = cPickle.load(open('{1}/train_data_{0}.pik'.format(index,path_to_dataset)))	
	Y_tr = train_data['labels']
	X_tr = train_data['features']
	print X_tr.shape
	print Y_tr.shape

	

	num_train = X_tr.shape[1]
	num_test = len(X_te)
	len_samples = X_tr.shape[0]

	num_classes = int(np.max(Y_tr) - np.min(Y_tr) + 1)
	inputD = X_tr.shape[2]
	outputD = num_classes 

	print 'Number of classes ',outputD
	print 'Feature dimension ',inputD

	epochs = 200
	batch_size = num_train
	learning_rate_decay = 0.97
	decay_after = 5
	
	use_pretrained = False
	train_more = False

	global rnn
	if not use_pretrained:
		# Creating network layers
		layers = [TemporalInputFeatures(inputD),LSTM('tanh','sigmoid','orthogonal',4,512),softmax(num_classes)]

		trY = T.lmatrix()

		# Initializing network
		rnn = RNN(layers,softmax_loss,trY,1e-3)

		if not os.path.exists('{1}/{0}/'.format(index,path_to_checkpoints)):
			os.mkdir('{1}/{0}/'.format(index,path_to_checkpoints))

		# Fitting model
		rnn.fitModel(X_tr,Y_tr,1,'{1}/{0}/'.format(index,path_to_checkpoints),epochs,batch_size,learning_rate_decay,decay_after)
	else:
		checkpoint = sys.argv[3]
		# Prediction
		rnn = load('{2}/{0}/checkpoint.{1}'.format(index,checkpoint,path_to_checkpoints))
		if train_more:
			rnn.fitModel(X_tr,Y_tr,1,'{1}/{0}/'.format(index,path_to_checkpoints),epochs,batch_size,learning_rate_decay,decay_after)


	predictions = []
	errors = 0
	N = 0
	for xte,yte in zip(X_te,Y_te):
		prediction = rnn.predict_output(xte,OutputMaxProb)
		predictions.append(prediction)
		t = np.nonzero(yte-prediction)
		print t
		errors += len(t[0])
		N += yte.shape[0]
	cPickle.dump(predictions,open('{1}/prediction_{0}.pik'.format(index,path_to_dataset),'wb'))
	print 'error = {0}'.format(errors*1.0/N)
