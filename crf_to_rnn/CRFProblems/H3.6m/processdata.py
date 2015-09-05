import numpy as np
import copy
from neuralmodels.utils import readCSVasFloat

trainSubjects = ['S1','S6','S7','S8','S9']
validateSubject = ['S11']
testSubject = ['S5']
allSubjects = ['S1','S6','S7','S8','S9','S11','S5']

actions =['directions','discussion','eating','greeting','phoning','posing','purchases','sitting','sittingdown','smoking','takingphoto','waiting','walking','walkingdog','walkingtogether']
subactions=['1','2']
path_to_dataset = ''

nodeFeatures={}
nodeFeatures['torso'] = range(6)
nodeFeatures['torso'] = nodeFeatures['torso'].extend(range(36,51))
nodeFeatures['right_arm'] = range(75,99)
nodeFeatures['left_arm'] = range(51,75)
nodeFeatures['right_leg'] = range(6,21)
nodeFeatures['left_leg'] = range(21,36)


def normalizationStats(completeData):
	data_mean = np.mean(completeData,axis=0)
	data_std =  np.std(completeData,axis=0)
	dimensions_to_ignore = list(np.where(data_std < 1e-4)[0])
	data_std[dimensions_to_ignore] = 1.0

	'''Returns the mean of data, std, and dimensions with small std. Which we later ignore.	'''
	return data_mean,data_std,dimensions_to_ignore

def sampleTrainSequences(trainData,T=200,delta_shift=50):
	training_data = []
	for k in trainData.keys():
		data = trainData[k]
		start = 0
		end = T
		while end < data.shape[0]:
			reshape_data = data[start:end,:].reshape((T,1,data.shape[1]))
			if len(training_data) == 0:
				training_data = reshape_data
			else:
				training_data = np.concatenate((training_data,reshape_data),axis=1)
			start += delta_shift
			end += delta_shift

	meanTensor = data_mean.reshape((1,1,training_data.shape[2]))	
	meanTensor = np.repeat(meanTensor,training_data.shape[0],axis=0)
	meanTensor = np.repeat(meanTensor,training_data.shape[1],axis=1)
	stdTensor = data_std.reshape((1,1,training_data.shape[2]))	
	stdTensor = np.repeat(stdTensor,training_data.shape[0],axis=0)
	stdTensor = np.repeat(stdTensor,training_data.shape[1],axis=1)

	# Normalizing the training data features
	training_data = np.divide((training_data - meanTensor),stdTensor)

def getfeatures(nodeName,edgeType):
	if edgeType.split('_')[1] == 'input':
		return nodeFeatures[nodeName]
	
	features = []
	nodesConnectedTo = nodeConnections[nodeName]
	for nm in nodesConnectedTo:
		et1 = nodeList[nm] + '_' + nodeList[nodeName]
		et2 = nodeList[nodeName] + '_' + nodeList[nm]
		
		if et1 == et2:
			f1 = nodeFeatures[nodeName] 
			f2 = nodeFeatures[nm]

		if et1 == edgeType or et2 == edgeType:
			if len(features) == 0




def cherryPickNodeFeatures(data3DTensor):
	nodeFeatures = {}
	nodeNames = nodeFeaturesRanges.keys()
	for nm in nodeNames:
		filterList = []
		for x in nodeFeaturesRanges[nm]:
			if x not in dimensions_to_ignore:
				filterList.append(x)
		nodeFeatures[nm] = data3Dtensor[:,:,filterList]
		

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
				else
					completeData = np.append(completeData,trainData[(subj,action,subact)],axis=0)
	return trainData,completeData
