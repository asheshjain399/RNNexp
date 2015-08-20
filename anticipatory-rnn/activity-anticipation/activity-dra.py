import sys
import numpy as np
import theano
import os
from theano import tensor as T
from readData import sortActivities
from neuralmodels.utils import permute 
from neuralmodels.loadcheckpoint import *
from neuralmodels.costs import softmax_loss
from neuralmodels.models import * #RNN, SharedRNN, SharedRNNVectors, SharedRNNOutput
from neuralmodels.predictions import OutputMaxProb, OutputSampleFromDiscrete
from neuralmodels.layers import * #softmax, simpleRNN, OneHot, LSTM, TemporalInputFeatures,ConcatenateFeatures,ConcatenateVectors
import cPickle
import pdb
import socket as soc

def DRAmodel(nodeList,edgeList,edgeFeatures,nodeFeatures,nodeToEdgeConnections):
	edgeRNNs = {}
	edgeNames = edgeList
	lstm_init = 'orthogonal'
	softmax_init = 'uniform'

	for em in edgeNames:
		inputJointFeatures = edgeFeatures[em]
		edgeRNNs[em] = [TemporalInputFeatures(inputJointFeatures),LSTM('tanh','sigmoid',lstm_init,4,128)]

	nodeRNNs = {}
	nodeNames = nodeList.keys()
	nodeLabels = {}
	for nm in nodeNames:
		num_classes = nodeList[nm]
		nodeRNNs[nm] = [LSTM('tanh','sigmoid',lstm_init,4,256),softmax(num_classes,softmax_init)]
		em = nm+'_input'
		edgeRNNs[em] = [TemporalInputFeatures(nodeFeatures[nm])]
		nodeLabels[nm] = T.lmatrix()
	dra = DRA(edgeRNNs,nodeRNNs,nodeToEdgeConnections,softmax_loss,nodeLabels,1e-3)
	return dra

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

	print "Loading training data...."
	train_data = cPickle.load(open('{1}/train_data_{0}.pik'.format(index,path_to_dataset)))	
	print "Data Loaded"
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

	assert(X_tr_human_shared.shape[0] == X_tr_human_disjoint.shape[0])
	assert(X_tr_human_shared.shape[1] == X_tr_human_disjoint.shape[1])
	assert(X_tr_objects_shared.shape[0] == X_tr_objects_disjoint.shape[0])
	assert(X_tr_objects_shared.shape[1] == X_tr_objects_disjoint.shape[1])


	nodeList = {}
	nodeList['H'] = num_sub_activities
	nodeList['O'] = num_affordances
	edgeList = ['HO']
	edgeFeatures = {}
	edgeFeatures['HO'] = inputJointFeatures
	nodeFeatures = {}
	nodeFeatures['H'] = inputHumanFeatures
	nodeFeatures['O'] = inputObjectFeatures
	nodeToEdgeConnections = {}
	nodeToEdgeConnections['H'] = {}
	nodeToEdgeConnections['H']['HO'] = [0,inputJointFeatures]
	nodeToEdgeConnections['H']['H_input'] = [inputJointFeatures,inputJointFeatures+inputHumanFeatures]
	nodeToEdgeConnections['O'] = {}
	nodeToEdgeConnections['O']['HO'] = [0,inputJointFeatures]
	nodeToEdgeConnections['O']['O_input'] = [inputJointFeatures,inputJointFeatures+inputObjectFeatures]
	dra = DRAmodel(nodeList,edgeList,edgeFeatures,nodeFeatures,nodeToEdgeConnections)

	trX = {}
	trY = {}
	trX['H'] = np.concatenate((X_tr_human_shared,X_tr_human_disjoint),axis=2)	
	trY['H'] = Y_tr_human
	trX['O'] = np.concatenate((X_tr_objects_shared,X_tr_objects_disjoint),axis=2)	
	trY['O'] = Y_tr_objects
	dra.fitModel(trX,trY,1,'{1}/{0}/'.format(index,path_to_checkpoints),10)
