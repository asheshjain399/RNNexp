import numpy as np
import copy
from neuralmodels.utils import readCSVasFloat
import socket as soc
import cPickle

global rng
rng = np.random.RandomState(1234567890)

trainSubjects = ['S1','S6','S7','S8','S9']
validateSubject = ['S11']
testSubject = ['S5']
allSubjects = ['S1','S6','S7','S8','S9','S11','S5']

#actions =['directions','discussion','eating','greeting','phoning','posing','purchases','sitting','sittingdown','smoking','takingphoto','waiting','walking','walkingdog','walkingtogether']
actions = ['walking','eating','smoking']
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
	dimensions_to_ignore = [0,1,2,3,4,5]
	dimensions_to_ignore.extend(list(np.where(data_std < 1e-4)[0]))
	data_std[dimensions_to_ignore] = 1.0

	'''Returns the mean of data, std, and dimensions with small std. Which we later ignore.	'''
	return data_mean,data_std,dimensions_to_ignore

def normalizeTensor(inputTensor):
	meanTensor = data_mean.reshape((1,1,inputTensor.shape[2]))	
	meanTensor = np.repeat(meanTensor,inputTensor.shape[0],axis=0)
	meanTensor = np.repeat(meanTensor,inputTensor.shape[1],axis=1)
	stdTensor = data_std.reshape((1,1,inputTensor.shape[2]))	
	stdTensor = np.repeat(stdTensor,inputTensor.shape[0],axis=0)
	stdTensor = np.repeat(stdTensor,inputTensor.shape[1],axis=1)
	normalizedTensor = np.divide((inputTensor - meanTensor),stdTensor)
	return normalizedTensor

def sampleTrainSequences(trainData,T=200,delta_shift=50):
	training_data = []
	Y = []
	N = 0
	for k in trainData.keys():

		'''Subsample data'''
		if len(k) < 4:
			continue

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
	data3Dtensor = normalizeTensor(data3Dtensor) #np.divide((data3Dtensor - meanTensor),stdTensor)
	Y3Dtensor = normalizeTensor(Y3Dtensor) #np.divide((Y3Dtensor - meanTensor),stdTensor)
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

def loadTrainData(subjects):
	trainData = {}
	completeData = []
	for subj in subjects:
		for action in actions:
			for subact in subactions:
				filename = '{0}/{1}/{2}_{3}.txt'.format(path_to_dataset,subj,action,subact)
				action_sequence = readCSVasFloat(filename)
				
				T = action_sequence.shape[0]
				odd_list = range(1,T,2)
				even_list = range(0,T,2)
				
				trainData[(subj,action,subact)] = action_sequence
				trainData[(subj,action,subact,'even')] = action_sequence[even_list,:]
				trainData[(subj,action,subact,'odd')] = action_sequence[odd_list,:]
				if len(completeData) == 0:
					completeData = copy.deepcopy(trainData[(subj,action,subact)])
				else:
					completeData = np.append(completeData,trainData[(subj,action,subact)],axis=0)
	return trainData,completeData

def generateForecastingExamples(trainData,prefix,suffix,subject):
	N = len(actions)*len(subactions)
	D = trainData[(subject,actions[0],subactions[0])].shape[1]
	trX = np.zeros((prefix,N,D),dtype=np.float32)
	trY = np.zeros((suffix,N,D),dtype=np.float32)
	count = 0
	for action in actions:
		for subact in subactions:
			T = trainData[(subject,action,subact,'even')].shape[0]
			idx = rng.randint(T-prefix-suffix)
			trX[:,count,:] = trainData[(subject,action,subact)][idx:(idx+prefix),:]
			trY[:,count,:] = trainData[(subject,action,subact)][(idx+prefix):(idx+prefix+suffix),:]
			count += 1
	return normalizeTensor(trX),normalizeTensor(trY)

def getMalikFeatures():
	return malikTrainFeatures,malikPredictFeatures

def getMalikValidationFeatures():
	return validate_malikTrainFeatures,validate_malikPredictFeatures

def getMalikTrajectoryForecasting():
	return trX_forecast_malik,trY_forecast_malik
	
#Keep T fixed, and tweak delta_shift in order to generate less/more examples
T=500
delta_shift=450

#Load training and validation data
[trainData,completeData]=loadTrainData(trainSubjects)
[validateData,completeValidationData]=loadTrainData(validateSubject)

#Compute training data mean
[data_mean,data_std,dimensions_to_ignore]=normalizationStats(completeData)
data_stats = {}
data_stats['mean'] = data_mean
data_stats['std'] = data_std
data_stats['ignore_dimensions'] = dimensions_to_ignore

#Create normalized 3D tensor for training and validation
[data3Dtensor,Y3Dtensor] = sampleTrainSequences(trainData,T,delta_shift)
[validate3Dtensor,validateY3Dtensor] = sampleTrainSequences(validateData,T,delta_shift)

print 'Training data stats (T,N,D) is ',data3Dtensor.shape
print 'Training data stats (T,N,D) is ',validate3Dtensor.shape

#Generate normalized data for trajectory forecasting
motion_prefix=50
motion_suffix=100
trX_forecast,trY_forecast = generateForecastingExamples(validateData,motion_prefix,motion_suffix,validateSubject[0])

#Create training and validation features for DRA
nodeFeatures = cherryPickNodeFeatures(data3Dtensor)
predictFeatures = cherryPickNodeFeatures(Y3Dtensor)
validate_nodeFeatures = cherryPickNodeFeatures(validate3Dtensor)
validate_predictFeatures = cherryPickNodeFeatures(validateY3Dtensor)

#Create training and validation features for Malik's LSTM model
malikTrainFeatures = ignoreZeroVarianceFeatures(data3Dtensor)
malikPredictFeatures = ignoreZeroVarianceFeatures(Y3Dtensor)
validate_malikTrainFeatures = ignoreZeroVarianceFeatures(validate3Dtensor)
validate_malikPredictFeatures = ignoreZeroVarianceFeatures(validateY3Dtensor)
trX_forecast_malik = ignoreZeroVarianceFeatures(trX_forecast)
trY_forecast_malik = ignoreZeroVarianceFeatures(trY_forecast)
