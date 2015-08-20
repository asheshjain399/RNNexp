import sys
import numpy as np
import theano
import os
from theano import tensor as T
from readData import sortActivities
from neuralmodels.utils import permute
from neuralmodels.costs import softmax_loss
from neuralmodels.models import * #RNN, SharedRNN, SharedRNNVectors, SharedRNNOutput
from neuralmodels.predictions import OutputMaxProb, OutputSampleFromDiscrete
from neuralmodels.layers import softmax, simpleRNN, OneHot, LSTM, TemporalInputFeatures,ConcatenateFeatures,ConcatenateVectors
import cPickle
import pdb
import socket as soc

#import activity_prediction_sharedrnn as prediction_met
'''
Joint anticipation OR detection for both human and objects
'''

def jointModel(num_sub_activities, num_affordances, inputJointFeatures,
	inputHumanFeatures,inputObjectFeatures):
	lstm_init = 'orthogonal'
	softmax_init = 'uniform'

	shared_input_layer = TemporalInputFeatures(inputJointFeatures)
	shared_hidden_layer = LSTM('tanh','sigmoid',lstm_init,4,128)
	#shared_hidden_layer = simpleRNN('tanh','orthogonal',4,128)
	shared_layers = [shared_input_layer,shared_hidden_layer]
	human_layers = [ConcatenateFeatures(inputHumanFeatures),LSTM('tanh','sigmoid',lstm_init,4,256),softmax(num_sub_activities,softmax_init)]
	object_layers = [ConcatenateFeatures(inputObjectFeatures),LSTM('tanh','sigmoid',lstm_init,4,256),softmax(num_affordances,softmax_init)]

	trY_1 = T.lmatrix()
	trY_2 = T.lmatrix()
	sharedrnn = SharedRNN(shared_layers,human_layers,object_layers,softmax_loss,trY_1,trY_2,1e-3)
	return sharedrnn

def jointModelVectors(num_sub_activities, num_affordances, inputJointFeatures,
		inputHumanFeatures,inputObjectFeatures):
	shared_input_layer = TemporalInputFeatures(inputJointFeatures)
	shared_hidden_layer = LSTM('tanh','sigmoid','orthogonal',4,128)
	shared_layers = [shared_input_layer,shared_hidden_layer]
	
	human_layers = [TemporalInputFeatures(inputHumanFeatures),LSTM('tanh','sigmoid','orthogonal',4,256)]
	human_activity_classification = [ConcatenateVectors(),softmax(num_sub_activities)]
	
	object_layers = [TemporalInputFeatures(inputObjectFeatures),LSTM('tanh','sigmoid','orthogonal',4,256)]
	object_affordance_classification = [ConcatenateVectors(),softmax(num_affordances)]

	trY_1 = T.lmatrix()
	trY_2 = T.lmatrix()
	sharedrnn = SharedRNNVectors(
				shared_layers, human_layers, object_layers, 
				human_activity_classification, object_affordance_classification, 
				softmax_loss, trY_1, trY_2, 1e-3
				)
	return sharedrnn


'''
Joint anticipation and detection for both human and objects
'''
def jointModelOutput(num_sub_activities, num_affordances, num_sub_activities_anticipation, 
		num_affordances_anticipation, inputJointFeatures, inputHumanFeatures, inputObjectFeatures):

	shared_input_layer = TemporalInputFeatures(inputJointFeatures)
	shared_hidden_layer = LSTM('tanh','sigmoid','orthogonal',4,128)
	#shared_hidden_layer = simpleRNN('tanh','orthogonal',4,128)
	shared_layers = [shared_input_layer,shared_hidden_layer]
	human_layers = [ConcatenateFeatures(inputHumanFeatures),LSTM('tanh','sigmoid','orthogonal',4,256)]
	object_layers = [ConcatenateFeatures(inputObjectFeatures),LSTM('tanh','sigmoid','orthogonal',4,256)]

	human_anticipation = [softmax(num_sub_activities_anticipation)]
	human_detection = [softmax(num_sub_activities)]

	object_anticipation = [softmax(num_affordances_anticipation)]
	object_detection = [softmax(num_affordances)]

	trY_1_detection = T.lmatrix()
	trY_2_detection = T.lmatrix()
	trY_1_anticipation = T.lmatrix()
	trY_2_anticipation = T.lmatrix()
	sharedrnn = SharedRNNOutput(
				shared_layers, human_layers, object_layers, 
				human_detection, human_anticipation, object_detection,
				object_anticipation, softmax_loss, trY_1_detection, 
				trY_2_detection,trY_1_anticipation,trY_2_anticipation,1e-3
				)
	return sharedrnn

if __name__ == '__main__':
	index = sys.argv[1]	
	fold = sys.argv[2]
	
	main_path = ''
	if soc.gethostname() == "napoli110.stanford.edu":
		main_path = '/scr/ashesh/activity-anticipation'
	elif soc.gethostname() == "ashesh":
		main_path = '.'
	
	path_to_dataset = '{1}/dataset/{0}'.format(fold,main_path)
	path_to_checkpoints = '{1}/checkpoints/{0}'.format(fold,main_path)

	if not os.path.exists(path_to_checkpoints):
		os.mkdir(path_to_checkpoints)

	test_data = cPickle.load(open('{1}/test_data_{0}.pik'.format(index,path_to_dataset)))	
	Y_te_human = test_data['labels_human']
	Y_te_human_anticipation = test_data['labels_human_anticipation']
	X_te_human_disjoint = test_data['features_human_disjoint']
	X_te_human_shared = test_data['features_human_shared']

	train_data = cPickle.load(open('{1}/train_data_{0}.pik'.format(index,path_to_dataset)))	
	Y_tr_human = train_data['labels_human']
	Y_tr_human_anticipation = train_data['labels_human_anticipation']
	X_tr_human_disjoint = train_data['features_human_disjoint']
	X_tr_human_shared = train_data['features_human_shared']

	Y_tr_objects = train_data['labels_objects']
	Y_tr_objects_anticipation = train_data['labels_objects_anticipation']
	X_tr_objects_disjoint = train_data['features_objects_disjoint']
	X_tr_objects_shared = train_data['features_objects_shared']

	num_sub_activities = int(np.max(Y_tr_human) - np.min(Y_tr_human) + 1)
	num_affordances = int(np.max(Y_tr_objects) - np.min(Y_tr_objects) + 1)
	num_sub_activities_anticipation = int(np.max(Y_tr_human_anticipation) - np.min(Y_tr_human_anticipation) + 1)
	num_affordances_anticipation = int(np.max(Y_tr_objects_anticipation) - np.min(Y_tr_objects_anticipation) + 1)
	inputJointFeatures = X_tr_human_shared.shape[2]
	inputHumanFeatures = X_tr_human_disjoint.shape[2]
	inputObjectFeatures = X_tr_objects_disjoint.shape[2]
	assert(inputJointFeatures == X_tr_objects_shared.shape[2])
	
	print '#human sub-activities ',num_sub_activities
	print '#object affordances ',num_affordances
	print '#human sub-activities-anticipation ',num_sub_activities_anticipation
	print '#object affordances_anticipation ',num_affordances_anticipation
	print 'shared features dim ',inputJointFeatures
	print 'human features dim ',inputHumanFeatures
	print 'object features dim ',inputObjectFeatures

	epochs = 10
	batch_size = X_tr_human_disjoint.shape[1]
	learning_rate_decay = 0.97
	decay_after = 5
	
	use_pretrained = False
	train_more = False

	architectures = ['detection','anticipation','joint']

	train_for = 0
	#train_for = 'current_prediction'	

	global rnn
	if not use_pretrained:
		if not os.path.exists('{1}/{0}/'.format(index,path_to_checkpoints)):
			os.mkdir('{1}/{0}/'.format(index,path_to_checkpoints))

		if architectures[train_for] == 'detection':

			rnn = jointModel(num_sub_activities, num_affordances, inputJointFeatures, 
					inputHumanFeatures, inputObjectFeatures )	
			rnn.fitModel(X_tr_human_shared, X_tr_human_disjoint, Y_tr_human, X_tr_objects_shared, 
				X_tr_objects_disjoint, Y_tr_objects, 1, '{1}/{0}/'.format(index,path_to_checkpoints), epochs,
				batch_size, learning_rate_decay, decay_after)

		elif architectures[train_for] == 'anticipation':

 			rnn = jointModel(num_sub_activities_anticipation, num_affordances_anticipation, 
					inputJointFeatures, inputHumanFeatures, inputObjectFeatures)	
			rnn.fitModel(X_tr_human_shared, X_tr_human_disjoint, Y_tr_human_anticipation,
				X_tr_objects_shared, X_tr_objects_disjoint, Y_tr_objects_anticipation, 1,
				'{1}/{0}/'.format(index,path_to_checkpoints), epochs, batch_size, learning_rate_decay, decay_after)

		elif architectures[train_for] == 'joint':

			rnn = jointModelOutput(num_sub_activities, num_affordances, num_sub_activities_anticipation, 
					num_affordances_anticipation, inputJointFeatures, inputHumanFeatures, inputObjectFeatures)	
			rnn.fitModel(X_tr_human_shared, X_tr_human_disjoint, Y_tr_human, Y_tr_human_anticipation,
				X_tr_objects_shared, X_tr_objects_disjoint, Y_tr_objects, Y_tr_objects_anticipation, 1,
				'{1}/{0}/'.format(index,path_to_checkpoints), epochs, batch_size, learning_rate_decay, decay_after)
		else:
			print "No training algorithm matched"
	else:
		checkpoint = sys.argv[3]
		# Prediction
		rnn = load('{2}/{0}/checkpoint.{1}'.format(index,checkpoint,path_to_checkpoints))
		if train_more:
			rnn.fitModel(X_tr,Y_tr,1,'{1}/{0}/'.format(index,path_to_checkpoints),epochs,batch_size,learning_rate_decay,decay_after)

	'''
	predictions = []
	errors = 0
	N = 0

	Y_te = Y_te_human_anticipation
	if train_for == 'current_prediction':
		Y_te = Y_te_human

	for xte_shared,xte,yte in zip(X_te_human_shared,X_te_human_disjoint,Y_te):
		if train_for == 'current_prediction':
			prediction = rnn.predict_output(xte_shared,xte,OutputMaxProb)
			predictions.append(prediction)
			t = np.nonzero(yte-prediction)
			print t
			errors += len(t[0])
			N += yte.shape[0]
		else:
			prediction = rnn.predict_output(xte_shared,xte,OutputMaxProb)
			predictions.append(prediction)
			t = np.nonzero(yte[:-1]-prediction[:-1])
			print t
			errors += len(t[0])
			N += yte.shape[0]-1

	cPickle.dump(predictions,open('{1}/prediction_{0}.pik'.format(index,path_to_dataset),'wb'))
	print 'acc = {0}'.format(1.0 - (errors*1.0/N))
	'''
