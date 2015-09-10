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

epoch_to_load = None
use_pretrained = 0
if len(sys.argv) > 2:
	use_pretrained = int(sys.argv[2])
	if len(sys.argv) <= 3:
		print 'enter the epoch to load as well'
		sys.exit(0)
	epoch_to_load = int(sys.argv[3])
	print 'Loading pre-trained model with epoch={0}'.format(epoch_to_load)

'''Hyperparameters'''
decay_type = 'schedule'
decay_after = -1
learning_rate_decay = 0.97
decay_schedule = [15,25,40,50]
decay_rate_schedule = [0.1,0.1,0.1,0.1]
lstm_size= 1000
lstm_init = 'orthogonal'
fc_init = 'uniform'
snapshot_rate = 10
epochs = 100
batch_size = 20
clipnorm = 0.0
use_noise = True
noise_schedule = [10,15,20,30]
noise_rate_schedule = [0.0001,0.001,0.01,0.1]

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

def saveForecastedMotion(forecast,path):
	T = forecast.shape[0]
	N = forecast.shape[1]
	D = forecast.shape[2]
	for j in range(N):
		motion = forecast[:,j,:]
		f = open('{0}ground_truth_forecast_N_{1}'.format(path,j),'w')
		for i in range(T):
			st = ''
			for k in range(D):
				st += str(motion[i,k]) + ','
			st = st[:-1]
			f.write(st+'\n')
		f.close()


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

def MaliksRegression(inputDim):
	layers = [TemporalInputFeatures(inputDim),
		#AddNoiseToInput(rng=rng),
		simpleRNN('rectify',fc_init,truncate_gradient=1,size=500,temporal_connection=False,rng=rng),
		simpleRNN('linear',fc_init,truncate_gradient=1,size=500,temporal_connection=False,rng=rng,jump_up=True),
		LSTM('tanh','sigmoid',lstm_init,truncate_gradient=50,size=lstm_size,rng=rng),
		LSTM('tanh','sigmoid',lstm_init,truncate_gradient=50,size=lstm_size,rng=rng,skip_input=True),		
		simpleRNN('rectify',fc_init,truncate_gradient=1,size=500,temporal_connection=False,rng=rng),
		simpleRNN('rectify',fc_init,truncate_gradient=1,size=100,temporal_connection=False,rng=rng),
		simpleRNN('linear',fc_init,truncate_gradient=1,size=inputDim,temporal_connection=False,rng=rng)
		]
	Y = T.tensor3(dtype=theano.config.floatX)
	learning_rate = T.scalar(dtype=theano.config.floatX)
	rnn = noisyRNN(layers,euclidean_loss,Y,learning_rate,clipnorm=clipnorm,update_type=Momentum())	
	return rnn

def LSTMRegression(inputDim):
	
	LSTMs = [LSTM('tanh','sigmoid',lstm_init,truncate_gradient=50,size=lstm_size,rng=rng,skip_input=False),
		LSTM('tanh','sigmoid',lstm_init,truncate_gradient=50,size=lstm_size,rng=rng,skip_input=True),		
		LSTM('tanh','sigmoid',lstm_init,truncate_gradient=50,size=lstm_size,rng=rng,skip_input=True)
		]
	layers = [TemporalInputFeatures(inputDim),
		AddNoiseToInput(rng=rng,jump_up=True),
		multilayerLSTM(LSTMs,skip_input=True,skip_output=True),
		simpleRNN('linear',fc_init,truncate_gradient=1,size=inputDim,temporal_connection=False,rng=rng)
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
	path_to_checkpoint = poseDataset.base_dir + '/checkpoints_Malik_s_{0}_lstm_init_{1}_fc_init_{2}_decay_type_{3}/'.format(lstm_size,lstm_init,fc_init,decay_type)

	if not os.path.exists(path_to_checkpoint):
		os.mkdir(path_to_checkpoint)
	saveNormalizationStats(path_to_checkpoint)

	trX,trY = poseDataset.getMalikFeatures()
	trX_validation,trY_validation = poseDataset.getMalikValidationFeatures()
	trX_forecasting,trY_forecasting = poseDataset.getMalikTrajectoryForecasting()
	
	saveForecastedMotion(trY_forecasting,path_to_checkpoint)
	print 'X forecasting ',trX_forecasting.shape
	print 'Y forecasting ',trY_forecasting.shape

	inputDim = trX.shape[2]
	rnn = MaliksRegression(inputDim)
	rnn.fitModel(trX,trY,snapshot_rate=shapshot_rate,path=path_to_checkpoint,epochs=epochs,batch_size=batch_size,
		decay_after=decay_after,learning_rate=1e-2,learning_rate_decay=learning_rate_decay,trX_validation=trX_validation,
		trY_validation=trY_validation,trX_forecasting=trX_forecasting,trY_forecasting=trY_forecasting,epoch_start=epoch_to_load,
		decay_type=decay_type,decay_schedule=decay_schedule,decay_rate_schedule=decay_rate_schedule,
		use_noise=use_noise,noise_schedule=noise_schedule,noise_rate_schedule=noise_rate_schedule)

def trainLSTM():
	path_to_checkpoint = poseDataset.base_dir + '/checkpoints_LSTM_no_Trans_no_rot_s_{0}_lstm_init_{1}_fc_init_{2}_decay_type_{3}/'.format(lstm_size,lstm_init,fc_init,decay_type)

	if not os.path.exists(path_to_checkpoint):
		os.mkdir(path_to_checkpoint)
	saveNormalizationStats(path_to_checkpoint)

	trX,trY = poseDataset.getMalikFeatures()
	trX_validation,trY_validation = poseDataset.getMalikValidationFeatures()
	trX_forecasting,trY_forecasting = poseDataset.getMalikTrajectoryForecasting()
	
	saveForecastedMotion(trY_forecasting,path_to_checkpoint)
	print 'X forecasting ',trX_forecasting.shape
	print 'Y forecasting ',trY_forecasting.shape

	inputDim = trX.shape[2]
	print inputDim
	rnn = []
	if use_pretrained == 1:
		rnn = load(path_to_checkpoint+'checkpoint.'+str(epoch_to_load))
	else:
		rnn = LSTMRegression(inputDim)
	rnn.fitModel(trX, trY, snapshot_rate=snapshot_rate, path=path_to_checkpoint, epochs=epochs, batch_size=batch_size,
		decay_after=decay_after, learning_rate=1e-2, learning_rate_decay=learning_rate_decay, trX_validation=trX_validation,
		trY_validation=trY_validation, trX_forecasting=trX_forecasting, trY_forecasting=trY_forecasting, epoch_start=epoch_to_load,
		decay_type=decay_type, decay_schedule=decay_schedule, decay_rate_schedule=decay_rate_schedule,
		use_noise=use_noise,noise_schedule=noise_schedule,noise_rate_schedule=noise_rate_schedule)

def saveNormalizationStats(path):
	cPickle.dump(poseDataset.data_stats,open('{0}h36mstats.pik'.format(path),'wb'))

if __name__ == '__main__':
	model_to_train = sys.argv[1]


	if model_to_train == 'malik':
		trainMaliks()
	elif model_to_train == 'dra':
		trainDRA()
	elif model_to_train == 'lstm':
		trainLSTM()
	else:
		print "Unknown model type ... existing"
