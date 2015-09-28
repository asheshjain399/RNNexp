import sys
import numpy as np
import theano
import os
from theano import tensor as T
from readData import sortActivities
from neuralmodels.utils import permute 
from neuralmodels.updates import *
from neuralmodels.loadcheckpoint import *
from neuralmodels.costs import softmax_loss
from neuralmodels.models import * #RNN, SharedRNN, SharedRNNVectors, SharedRNNOutput
from neuralmodels.predictions import OutputMaxProb, OutputSampleFromDiscrete
from neuralmodels.layers import * #softmax, simpleRNN, OneHot, LSTM, TemporalInputFeatures,ConcatenateFeatures,ConcatenateVectors
import cPickle
import pdb
import socket as soc

'''
Understanding data structures
nodeToEdgeConnections: node_type ---> [edge_types]
nodeConnections: node_name ---> [node_names]
nodeNames: node_name ---> node_type
nodeList: node_type ---> output_dimension
nodeFeatureLength: node_type ---> feature_dim_into_nodeRNN

edgeList: list of edge types
edgeFeatures: edge_type ---> feature_dim_into_edgeRNN
'''

def DRAmodel(nodeList,edgeList,edgeListComplete,edgeFeatures,nodeFeatures,nodeToEdgeConnections,clipnorm=25.0,train_for='joint'):
	edgeRNNs = {}
	edgeTypes = edgeList
	lstm_init = 'orthogonal'
	softmax_init = 'uniform'
	
	rng = np.random.RandomState(1234567890)

	for et in edgeTypes:
		inputJointFeatures = edgeFeatures[et]
		print inputJointFeatures
		edgeRNNs[et] = [TemporalInputFeatures(inputJointFeatures),LSTM('tanh','sigmoid',lstm_init,truncate_gradient=4,size=128,rng=rng)] #128

	nodeRNNs = {}
	nodeTypes = nodeList.keys()
	nodeLabels = {}
	outputLayer = {}
	for nt in nodeTypes:
		num_classes = nodeList[nt]
		#nodeRNNs[nt] = [LSTM('tanh','sigmoid',lstm_init,truncate_gradient=4,size=256,rng=rng),softmax(num_classes,softmax_init,rng=rng)] #256
		nodeRNNs[nt] = [LSTM('tanh','sigmoid',lstm_init,truncate_gradient=4,size=256,rng=rng)] #256
		if train_for=='joint':
			nodeLabels[nt] = {}
			nodeLabels[nt]['detection'] = T.lmatrix()
			nodeLabels[nt]['anticipation'] = T.lmatrix()
			outputLayer[nt] = [softmax(num_classes,softmax_init,rng=rng),softmax(num_classes+1,softmax_init,rng=rng)]
		else:
			nodeLabels[nt] = T.lmatrix()
			outputLayer[nt] = [softmax(num_classes,softmax_init,rng=rng)]
		et = nt+'_input'
		edgeRNNs[et] = [TemporalInputFeatures(nodeFeatures[nt])]
	learning_rate = T.fscalar()
	dra = DRAanticipation(edgeRNNs,nodeRNNs,outputLayer,nodeToEdgeConnections,edgeListComplete,softmax_loss,nodeLabels,learning_rate,clipnorm,train_for=train_for)
	return dra

if __name__ == '__main__':
	index = sys.argv[1]	
	fold = sys.argv[2]
	
	main_path = ''
	if soc.gethostname()[:6] == "napoli":
		main_path = '/scail/scratch/group/cvgl/ashesh/activity-anticipation'
	elif soc.gethostname() == "ashesh":
		main_path = '.'
			
	path_to_dataset = '{1}/dataset/{0}'.format(fold,main_path)
	path_to_checkpoints = '{1}/checkpoints/{0}'.format(fold,main_path)

	if not os.path.exists(path_to_checkpoints):
		os.mkdir(path_to_checkpoints)


	test_data = cPickle.load(open('{1}/test_data_{0}.pik'.format(index,path_to_dataset)))
	print 	'{1}/test_data_{0}.pik'.format(index,path_to_dataset)
	Y_te_human = test_data['labels_human']
	Y_te_human_anticipation = test_data['labels_human_anticipation']
	X_te_human_disjoint = test_data['features_human_disjoint']
	X_te_human_shared = test_data['features_human_shared']

	Y_te_objects = test_data['labels_objects']
	Y_te_objects_anticipation = test_data['labels_objects_anticipation']
	X_te_objects_disjoint = test_data['features_objects_disjoint']
	X_te_objects_shared = test_data['features_objects_shared']

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
	edgeCompleteList = ['HO','H_input','O_input']
	dra = DRAmodel(nodeList,edgeList,edgeCompleteList,edgeFeatures,nodeFeatures,nodeToEdgeConnections)

	trX = {}
	trY = {}
	trY['detection'] = {}
	trY['anticipation'] = {}
	trX['H:H'] = np.concatenate((X_tr_human_shared,X_tr_human_disjoint),axis=2)	
	trX['O:O'] = np.concatenate((X_tr_objects_shared,X_tr_objects_disjoint),axis=2)	
	trY['detection']['H:H'] = Y_tr_human
	trY['detection']['O:O'] = Y_tr_objects
	trY['anticipation']['H:H'] = Y_tr_human_anticipation
	trY['anticipation']['O:O'] = Y_tr_objects_anticipation

	trX_validation = {}
	trY_validation = {}
	trY_validation['detection'] = {}
	trY_validation['anticipation'] = {}

	trX_validation['H:H'] = []
	trX_validation['O:O'] = []
	trY_validation['detection']['H:H'] = []
	trY_validation['detection']['O:O'] = []
	trY_validation['anticipation']['H:H'] = []
	trY_validation['anticipation']['O:O'] = []


	for y_a,y,share,disjoint in zip(Y_te_human_anticipation,Y_te_human,X_te_human_shared,X_te_human_disjoint):
		trY_validation['detection']['H:H'].append(y)
		trY_validation['anticipation']['H:H'].append(y_a)
		trX_validation['H:H'].append(np.concatenate((share,disjoint),axis=2))
	for y_a,y,share,disjoint in zip(Y_te_objects_anticipation,Y_te_objects,X_te_objects_shared,X_te_objects_disjoint):
		trY_validation['detection']['O:O'].append(y)
		trY_validation['anticipation']['H:H'].append(y_a)
		trX_validation['O:O'].append(np.concatenate((share,disjoint),axis=2))

	checkpoint_dir = '{1}/{0}/'.format(index,path_to_checkpoints)
	if not os.path.exists(checkpoint_dir):
		os.mkdir(checkpoint_dir)
	dra.fitModel(trX,trY,10,checkpoint_dir,epochs=10,std=1e-5,trX_validation=trX_validation,trY_validation=trY_validation,predictfn=OutputMaxProb,maxiter=1000)
