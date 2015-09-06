import numpy as np
import copy
from neuralmodels.utils import readCSVasFloat
import socket as soc

trainSubjects = ['S1','S6','S7','S8','S9']
validateSubject = ['S11']
testSubject = ['S5']
allSubjects = ['S1','S6','S7','S8','S9','S11','S5']

actions =['directions','discussion','eating','greeting','phoning','posing','purchases','sitting','sittingdown','smoking','takingphoto','waiting','walking','walkingdog','walkingtogether']
subactions=['1','2']

base_dir = ''
if soc.gethostname() == "napoli110.stanford.edu":
	base_dir = '/scr/ashesh/h3.6m'
elif soc.gethostname() == "ashesh":
	base_dir = '.'
path_to_dataset = '{0}/dataset'.format(base_dir)

nodeFeaturesRanges={}
nodeFeaturesRanges['torso'] = range(6)
nodeFeaturesRanges['torso'].extend(range(36,51))
nodeFeaturesRanges['right_arm'] = range(75,99)
nodeFeaturesRanges['left_arm'] = range(51,75)
nodeFeaturesRanges['right_leg'] = range(6,21)
nodeFeaturesRanges['left_leg'] = range(21,36)


def normalizationStats(completeData):
	data_mean = np.mean(completeData,axis=0)
	data_std =  np.std(completeData,axis=0)
	dimensions_to_ignore = list(np.where(data_std < 1e-4)[0])
	data_std[dimensions_to_ignore] = 1.0

	'''Returns the mean of data, std, and dimensions with small std. Which we later ignore.	'''
	return data_mean,data_std,dimensions_to_ignore

def sampleTrainSequences(trainData,T=200,delta_shift=50):
	training_data = []
	Y = []
	N = 0
	for k in trainData.keys():
		data = trainData[k]
		start = 0
		end = T
		while end + 1 < data.shape[0]:
			training_data.append(data[start:end,:])
			Y.append(data[start+1:end+1,:])
			N += 1
			start += delta_shift
			end += delta_shift
	D = training_data[0].shape[1]
	data3Dtensor = np.zeros((T,N,D),dtype=np.float32)
	Y3Dtensor = np.zeros((T,N,D),dtype=np.float32)
	count = 0
	for x,y in zip(training_data,Y):
		data3Dtensor[:,count,:] = x
		Y3Dtensor[:,count,:] = y
		count += 1
	meanTensor = data_mean.reshape((1,1,data3Dtensor.shape[2]))	
	meanTensor = np.repeat(meanTensor,data3Dtensor.shape[0],axis=0)
	meanTensor = np.repeat(meanTensor,data3Dtensor.shape[1],axis=1)
	stdTensor = data_std.reshape((1,1,data3Dtensor.shape[2]))	
	stdTensor = np.repeat(stdTensor,data3Dtensor.shape[0],axis=0)
	stdTensor = np.repeat(stdTensor,data3Dtensor.shape[1],axis=1)

	# Normalizing the training data features
	data3Dtensor = np.divide((data3Dtensor - meanTensor),stdTensor)
	Y3Dtensor = np.divide((Y3Dtensor - meanTensor),stdTensor)
	return data3Dtensor,Y3Dtensor

def getlabels(nodeName):
	D = predictFeatures[nodeName].shape[2]
	return predictFeatures[nodeName],D

def getfeatures(nodeName,edgeType,nodeConnections,nodeNames):
	if edgeType.split('_')[1] == 'input':
		return nodeFeatures[nodeName]
	
	features = []
	nodesConnectedTo = nodeConnections[nodeName]
	for nm in nodesConnectedTo:
		et1 = nodeNames[nm] + '_' + nodeNames[nodeName]
		et2 = nodeNames[nodeName] + '_' + nodeNames[nm]
		
		f1 = 0
		f2 = 0
		if et1 == et2 and et1 == edgeType:
			f1 = nodeFeatures[nodeName] 
			f2 = nodeFeatures[nm]
		elif et1 == edgeType:
			f1 = nodeFeatures[nm] 
			f2 = nodeFeatures[nodeName]
		elif et2 == edgeType:
			f1 = nodeFeatures[nodeName] 
			f2 = nodeFeatures[nm]
		else:
			continue

		if len(features) == 0:
			features = np.concatenate((f1,f2),axis=2)
		else:
			features += np.concatenate((f1,f2),axis=2)

	return features	

def cherryPickNodeFeatures(data3DTensor):
	nodeFeatures = {}
	nodeNames = nodeFeaturesRanges.keys()
	for nm in nodeNames:
		filterList = []
		for x in nodeFeaturesRanges[nm]:
			if x not in dimensions_to_ignore:
				filterList.append(x)
		nodeFeatures[nm] = data3Dtensor[:,:,filterList]
	return nodeFeatures	

def ignoreZeroVarianceFeatures(data3DTensor):
	D = data3DTensor.shape[2]
	filterList = []
	for x in range(D):
		if x in dimensions_to_ignore:
			continue
		filterList.append(x)
	return data3DTensor[:,:,filterList]

def loadTrainData():
	trainData = {}
	completeData = []
	for subj in trainSubjects:
		for action in actions:
			for subact in subactions:
				filename = '{0}/{1}/{2}_{3}.txt'.format(path_to_dataset,subj,action,subact)
				trainData[(subj,action,subact)] = readCSVasFloat(filename)
				if len(completeData) == 0:
					completeData = copy.deepcopy(trainData[(subj,action,subact)])
				else:
					completeData = np.append(completeData,trainData[(subj,action,subact)],axis=0)
	return trainData,completeData

def getMalikFeatures():
	return malikTrainFeatures,malikPredictFeatures

[trainData,completeData]=loadTrainData()

[data_mean,data_std,dimensions_to_ignore]=normalizationStats(completeData)

[data3Dtensor,Y3Dtensor] = sampleTrainSequences(trainData,T=200,delta_shift=50)

nodeFeatures = cherryPickNodeFeatures(data3Dtensor)

predictFeatures = cherryPickNodeFeatures(Y3Dtensor)

malikTrainFeatures = ignoreZeroVarianceFeatures(data3Dtensor)

malikPredictFeatures = ignoreZeroVarianceFeatures(Y3Dtensor)
