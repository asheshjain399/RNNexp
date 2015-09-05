import sys
import numpy as np
import theano
import os
from theano import tensor as T
from neuralmodels.utils import permute 
from neuralmodels.loadcheckpoint import *
from neuralmodels.costs import softmax_loss, euclidean_loss
from neuralmodels.models import * 
from neuralmodels.predictions import OutputMaxProb, OutputSampleFromDiscrete
from neuralmodels.layers import * 
import cPickle
import pdb
import socket as soc
import copy

sys.path.insert('CRFProblems/H3.6m')
import processdata as poseDataset

global rng
rng = np.random.RandomState(1234567890)


def DRAmodelRegression(nodeList,edgeList,edgeFeatures,nodeFeatureLength,nodeToEdgeConnections,clipnorm=0.0):

	edgeRNNs = {}
	edgeNames = edgeList
	lstm_init = 'orthogonal'

	for em in edgeNames:
		inputJointFeatures = edgeFeatures[em]
		edgeRNNs[em] = [TemporalInputFeatures(inputJointFeatures),
				AddNoiseToInput(rng=rng),
				simpleRNN('rectify','uniform',size=500,temporal_connection=False,rng=rng),
				simpleRNN('linear','uniform',size=500,temporal_connection=False,rng=rng),
				LSTM('tanh','sigmoid',lstm_init,100,1000,rng=rng),
				LSTM('tanh','sigmoid',lstm_init,100,1000,rng=rng)
				]

	nodeRNNs = {}
	nodeTypes = nodeList.keys()
	nodeLabels = {}
	for nm in nodeTypes:
		num_classes = nodeList[nm]
		nodeRNNs[nm] = [LSTM('tanh','sigmoid',lstm_init,100,1000,rng=rng),
				simpleRNN('rectify','uniform',size=500,temporal_connection=False,rng=rng),
				simpleRNN('rectify','uniform',size=100,temporal_connection=False,rng=rng),
				simpleRNN('linear','uniform',size=num_classes,temporal_connection=False,rng=rng),
				]
		em = nm+'_input'
		edgeRNNs[em] = [TemporalInputFeatures(nodeFeatureLength[nm]),
				AddNoiseToInput(rng=rng),
				simpleRNN('rectify','uniform',size=500,temporal_connection=False,rng=rng),
				simpleRNN('linear','uniform',size=500,temporal_connection=False,rng=rng),
				]
		nodeLabels[nm] = T.tensor3(dtype=theano.config.floatX)
	learning_rate = T.scalar(dtype=theano.config.floatX)
	dra = DRA(edgeRNNs,nodeRNNs,nodeToEdgeConnections,euclidean_loss,nodeLabels,learning_rate,clipnorm)
	return dra

def DRAmodelClassification(nodeList,edgeList,edgeFeatures,nodeFeatureLength,nodeToEdgeConnections,clipnorm=0.0):

	edgeRNNs = {}
	edgeNames = edgeList
	lstm_init = 'orthogonal'
	softmax_init = 'uniform'

	for em in edgeNames:
		inputJointFeatures = edgeFeatures[em]
		edgeRNNs[em] = [TemporalInputFeatures(inputJointFeatures),
				AddNoiseToInput(rng=rng),
				simpleRNN('rectify','uniform',size=500,temporal_connection=False,rng=rng),
				simpleRNN('rectify','uniform',size=500,temporal_connection=False,rng=rng),
				LSTM('tanh','sigmoid',lstm_init,100,1000,rng=rng),
				LSTM('tanh','sigmoid',lstm_init,100,1000,rng=rng)
				]

	nodeRNNs = {}
	nodeNames = nodeList.keys()
	nodeLabels = {}
	for nm in nodeNames:
		num_classes = nodeList[nm]
		nodeRNNs[nm] = [LSTM('tanh','sigmoid',lstm_init,100,1000,rng=rng),
				simpleRNN('rectify','uniform',size=500,temporal_connection=False,rng=rng),
				simpleRNN('rectify','uniform',size=100,temporal_connection=False,rng=rng),
				simpleRNN('rectify','uniform',size=54,temporal_connection=False,rng=rng),
				softmax(num_classes,softmax_init,rng=rng)
				]
		em = nm+'_input'
		edgeRNNs[em] = [TemporalInputFeatures(nodeFeatureLength[nm]),
				AddNoiseToInput(rng=rng),
				simpleRNN('rectify','uniform',size=500,temporal_connection=False,rng=rng),
				simpleRNN('rectify','uniform',size=500,temporal_connection=False,rng=rng),
				]
		nodeLabels[nm] = T.lmatrix()
	learning_rate = T.fscalar()
	dra = DRA(edgeRNNs,nodeRNNs,nodeToEdgeConnections,softmax_loss,nodeLabels,learning_rate,clipnorm)
	return dra

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

def readCRFGraph(filename):
	lines = open(filename).readlines()
	nodeOrder = []
	nodeNames = {}
	nodeList = {}
	nodeToEdgeConnections = {}
	nodeFeatureLength = {}
	for node_name, node_type in zip(lines[0].strip().split(','),lines[1].strip().split(',')):
		nodeOrder.append(node_name)
		nodeNames[node_name] = node_type
		nodeList[node_type] = 0
		nodeToEdgeConnections[node_type] = {}
		nodeToEdgeConnections[node_type][node_type+'_input'] = [0,0]
		nodeFeatureLength[node_type] = 0
	
	edgeList = []
	edgeFeatures = {}
	nodeConnections = {}
	for i in range(2,len(lines)):
		first_nodeName = nodeOrder[i-2]
		first_nodeType = nodeNames[first_nodeName]
		nodeConnections[first_nodeName] = []
		connections = lines[i].strip().split(',')
		for j in range(len(connections)):
			if connections[j] == '1':
				second_nodeName = nodeOrder[j]
				second_nodeType = nodeNames[second_nodeName]
				nodeConnections[first_nodeName].append(second_nodeName)
		
				edgeType_1 = first_nodeType + '_' + second_nodeType
				edgeType_2 = second_nodeType + '_' + first_nodeType
				edgeType = ''
				if edgeType_1 in edgeList:
					edgeType = edgeType_1
					continue
				elif edgeType_2 in edgeList:
					edgeType = edgeType_2
					continue
				else:
					edgeType = edgeType_1
				edgeList.append(edgeType)
				edgeFeatures[edgeType] = 0
				nodeToEdgeConnections[first_nodeType][edgeType] = [0,0]
				nodeToEdgeConnections[second_nodeType][edgeType] = [0,0]

	trX = {}
	trY = {}
	for nodeType in nodeList:
		trX[nodeType] = {}
		trY[nodeType] = {}

	for nodeName in nodeNames.keys():
		edge_features = {}
		nodeType = nodeNames[nodeName]
		edgeTypesConnectedTo = nodeToEdgeConnections[nodeType].keys()
		low = 0
		high = 0

		for edgeType in edgeTypesConnectedTo:
			edge_features[edgeType] = poseDataset.getfeatures(nodeName,edgeType,nodeConnections,nodeNames)

		edgeType = nodeType + '_input'
		D = edge_features[edgeType].shape[2]
		nodeFeatureLength[nodeType] = D
		high += D
		nodeToEdgeConnections[nodeType][edgeType][0] = low
		nodeToEdgeConnections[nodeType][edgeType][1] = high
		low = high
		nodeRNNFeatures = copy.deepcopy(edge_features[edgeType])

		for edgeType in edgeList:
			if edgeType not in edgeTypesConnectedTo:
				continue
			D = edge_features[edgeType].shape[2]
			edgeFeatures[edgeType] = D
			high += D
			nodeToEdgeConnections[nodeType][edgeType][0] = low
			nodeToEdgeConnections[nodeType][edgeType][1] = high
			low = high
			nodeRNNFeatures = np.concatenate((nodeRNNFeatures,edge_features[edgeType]),axis=2)
		
		Y,num_classes = getLabels(nodeName)
		nodeList[nodeType] = num_classes
		
		trX[nodeType][nodeName] = nodeRNNFeatures
		trY[nodeType][nodeName] = Y

		print nodeToEdgeConnections

	return nodeNames,nodeList,nodeFeatureLength,nodeConnections,edgeList,edgeFeatures,nodeToEdgeConnections,trX,trY	

if __name__ == '__main__':
	
	crf_problem = sys.argv[1]

	crf_file = './CRFProblems/{0}/crf'.format(crf_problem)

	[nodeNames,nodeList,nodeFeatureLength,nodeConnections,edgeList,edgeFeatures,nodeToEdgeConnections,trX,trY] = readCRFGraph(crf_file)
	dra = DRAmodelRegression(nodeList,edgeList,edgeFeatures,nodeFeatureLength,nodeToEdgeConnections,clipnorm=0.0)
	'''
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
	'''
