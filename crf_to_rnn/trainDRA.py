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
from neuralmodels.updates import Adagrad,RMSprop,Momentum,Adadelta
import cPickle
import pdb
import socket as soc
import copy


'''Loads H3.6m dataset'''
sys.path.insert(0,'CRFProblems/H3.6m')
import processdata as poseDataset
print '**** H3.6m Loaded ****'

global rng
rng = np.random.RandomState(1234567890)

def readCRFGraph(filename):
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
		
		Y,num_classes = poseDataset.getlabels(nodeName)
		nodeList[nodeType] = num_classes
		
		idx = nodeName + ':' + nodeType
		trX[idx] = nodeRNNFeatures
		trY[idx] = Y

		print nodeToEdgeConnections

	return nodeNames,nodeList,nodeFeatureLength,nodeConnections,edgeList,edgeFeatures,nodeToEdgeConnections,trX,trY	



def DRAmodelRegression(nodeList,edgeList,edgeFeatures,nodeFeatureLength,nodeToEdgeConnections,clipnorm=0.0):

	edgeRNNs = {}
	edgeNames = edgeList
	lstm_init = 'orthogonal'

	for em in edgeNames:
		inputJointFeatures = edgeFeatures[em]
		edgeRNNs[em] = [TemporalInputFeatures(inputJointFeatures),
				AddNoiseToInput(rng=rng),
				simpleRNN('rectify','uniform',truncate_gradient=1,size=500,temporal_connection=False,rng=rng),
				simpleRNN('linear','uniform',truncate_gradient=1,size=500,temporal_connection=False,rng=rng),
				LSTM('tanh','sigmoid',lstm_init,100,1000,rng=rng),
				LSTM('tanh','sigmoid',lstm_init,100,1000,rng=rng)
				]

	nodeRNNs = {}
	nodeTypes = nodeList.keys()
	nodeLabels = {}
	for nm in nodeTypes:
		num_classes = nodeList[nm]
		nodeRNNs[nm] = [LSTM('tanh','sigmoid',lstm_init,100,1000,rng=rng),
				simpleRNN('rectify','uniform',truncate_gradient=1,size=500,temporal_connection=False,rng=rng),
				simpleRNN('rectify','uniform',truncate_gradient=1,size=100,temporal_connection=False,rng=rng),
				simpleRNN('linear','uniform',truncate_gradient=1,size=num_classes,temporal_connection=False,rng=rng),
				]
		em = nm+'_input'
		edgeRNNs[em] = [TemporalInputFeatures(nodeFeatureLength[nm]),
				AddNoiseToInput(rng=rng),
				simpleRNN('rectify','uniform',truncate_gradient=1,size=500,temporal_connection=False,rng=rng),
				simpleRNN('linear','uniform',truncate_gradient=1,size=500,temporal_connection=False,rng=rng),
				]
		nodeLabels[nm] = T.tensor3(dtype=theano.config.floatX)
	learning_rate = T.scalar(dtype=theano.config.floatX)
	dra = DRA(edgeRNNs,nodeRNNs,nodeToEdgeConnections,euclidean_loss,nodeLabels,learning_rate,clipnorm=clipnorm)
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
	dra = DRA(edgeRNNs,nodeRNNs,nodeToEdgeConnections,softmax_loss,nodeLabels,learning_rate,clipnorm=clipnorm)
	return dra

def MaliksRegression(inputDim,clipnorm=0.0):
	lstm_init = 'orthogonal'
	layers = [TemporalInputFeatures(inputDim),
		AddNoiseToInput(rng=rng),
		simpleRNN('rectify','uniform',truncate_gradient=1,size=500,temporal_connection=False,rng=rng),
		simpleRNN('linear','uniform',truncate_gradient=1,size=500,temporal_connection=False,rng=rng),
		LSTM('tanh','sigmoid',lstm_init,truncate_gradient=50,size=1000,rng=rng),
		LSTM('tanh','sigmoid',lstm_init,truncate_gradient=50,size=1000,rng=rng),		
		simpleRNN('rectify','uniform',truncate_gradient=1,size=500,temporal_connection=False,rng=rng),
		simpleRNN('rectify','uniform',truncate_gradient=1,size=100,temporal_connection=False,rng=rng),
		simpleRNN('linear','uniform',truncate_gradient=1,size=inputDim,temporal_connection=False,rng=rng)
		]
	Y = T.tensor3(dtype=theano.config.floatX)
	learning_rate = T.scalar(dtype=theano.config.floatX)
	rnn = noisyRNN(layers,euclidean_loss,Y,learning_rate,clipnorm=clipnorm,update_type=Momentum())	
	return rnn

def trainDRA():
	crf_file = './CRFProblems/H3.6m/crf'
	path_to_checkpoint = poseDataset.base_dir + '/checkpoints_DRA/'
	[nodeNames,nodeList,nodeFeatureLength,nodeConnections,edgeList,edgeFeatures,nodeToEdgeConnections,trX,trY] = readCRFGraph(crf_file)
	dra = DRAmodelRegression(nodeList,edgeList,edgeFeatures,nodeFeatureLength,nodeToEdgeConnections,clipnorm=0.0)
	dra.fitModel(trX,trY,1,path=path_to_checkpoint,epochs=50,batch_size=200,decay_after=15)

def trainMaliks():
	path_to_checkpoint = poseDataset.base_dir + '/checkpoints_Malik/'

	trX,trY = poseDataset.getMalikFeatures()
	trX_validation,trY_validation = poseDataset.getMalikValidationFeatures()
	trX_forecasting,trY_forecasting = poseDataset.getMalikTrajectoryForecasting()

	inputDim = trX.shape[2]
	rnn = MaliksRegression(inputDim,clipnorm=25.0)
	rnn.fitModel(trX,trY,1,path=path_to_checkpoint,epochs=100,batch_size=20,decay_after=10,
		learning_rate=1e-2,trX_validation=trX_validation,trY_validation=trY_validation,
		trX_forecasting=trX_forecasting,trY_forecasting=trY_forecasting)
		
if __name__ == '__main__':
	model_to_train = sys.argv[1]

	if model_to_train == 'malik':
		trainMaliks()
	elif model_to_train == 'dra':
		trainDRA()
	else:
		print "Unknown model type ... existing"
