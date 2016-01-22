import numpy as np
import copy
from neuralmodels.utils import readCSVasFloat
import socket as soc
import cPickle

global rng
rng = np.random.RandomState(1234567890)

global trainSubjects,validateSubject,testSubject,actions

trainSubjects = ['S1','S6','S7','S8','S9']
validateSubject = ['S11']
testSubject = ['S5']
allSubjects = ['S1','S6','S7','S8','S9','S11','S5']

#actions =['directions','discussion','eating','greeting','phoning','posing','purchases','sitting','sittingdown','smoking','takingphoto','waiting','walking','walkingdog','walkingtogether']
actions = ['walking','eating','smoking']
subactions=['1','2']

base_dir = ''
base_dir = '/scail/scratch/group/cvgl/ashesh/h3.6m'

nodeFeaturesRanges={}
nodeFeaturesRanges['torso'] = range(6)
nodeFeaturesRanges['torso'].extend(range(36,51))
nodeFeaturesRanges['right_arm'] = range(75,99)
nodeFeaturesRanges['left_arm'] = range(51,75)
nodeFeaturesRanges['right_leg'] = range(6,21)
nodeFeaturesRanges['left_leg'] = range(21,36)
drop_right_knee = [9,10,11]

def normalizationStats(completeData):
	data_mean = np.mean(completeData,axis=0)
	data_std =  np.std(completeData,axis=0)
	dimensions_to_ignore = [] 
	if not full_skeleton:
		dimensions_to_ignore = [0,1,2,3,4,5]
	dimensions_to_ignore.extend(list(np.where(data_std < 1e-4)[0]))
	data_std[dimensions_to_ignore] = 1.0
	print dimensions_to_ignore
	new_idx = []
	count = 0
	for i in range(completeData.shape[1]):
		if i in dimensions_to_ignore:
			new_idx.append(-1)
		else:
			new_idx.append(count)
			count += 1

	'''Returns the mean of data, std, and dimensions with small std. Which we later ignore.	'''
	return data_mean,data_std,dimensions_to_ignore,np.array(new_idx)

def normalizeTensor(inputTensor):
	meanTensor = data_mean.reshape((1,1,inputTensor.shape[2]))	
	meanTensor = np.repeat(meanTensor,inputTensor.shape[0],axis=0)
	meanTensor = np.repeat(meanTensor,inputTensor.shape[1],axis=1)
	stdTensor = data_std.reshape((1,1,inputTensor.shape[2]))	
	stdTensor = np.repeat(stdTensor,inputTensor.shape[0],axis=0)
	stdTensor = np.repeat(stdTensor,inputTensor.shape[1],axis=1)
	normalizedTensor = np.divide((inputTensor - meanTensor),stdTensor)
	return normalizedTensor


def sampleConnectedTrainSequences(trainData,T=200,delta_shift=50):
	training_data = []
	Y = []
	N = 0
	start= 0
	end = T
	minibatch_size = 0

	training_keys = trainData.keys()
	for k in training_keys:
		if len(k) < 4:
			continue
		if not k[3] == 'even':
			continue
		minibatch_size += 1


	while(True):
		isEnd = True
		for k in training_keys:

			if len(k) < 4:
				continue
			if not k[3] == 'even':
				continue

			data = trainData[k]
			fae = np.zeros((T,data.shape[1]),dtype=np.float32)
			labels = np.zeros((T,data.shape[1]),dtype=np.float32)

			if end + 1 < data.shape[0]:
				isEnd = False
				fea = data[start:end,:]
				labels = data[start+1:end+1,:]
			training_data.append(fea)
			Y.append(labels)
			N += 1
		if isEnd:
			break
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
	return data3Dtensor,Y3Dtensor,minibatch_size


def sampleTrainSequences(trainData,T=200,delta_shift=50):
	training_data = []
	t_minus_one_data = []
	Y = []
	N = 0
	for k in trainData.keys():

		if len(k) == th_len:
			continue

		data = trainData[k]
		start = 16
		end = T + 16
		while end + 1 < data.shape[0]:
			training_data.append(data[start:end,:])
			t_minus_one_data.append(data[start-1:end-1,:])
			Y.append(data[start+1:end+1,:])
			N += 1
			start += delta_shift
			end += delta_shift
	D = training_data[0].shape[1]
	data3Dtensor = np.zeros((T,N,D),dtype=np.float32)
	data3Dtensor_t_1 = np.zeros((T,N,D),dtype=np.float32)
	Y3Dtensor = np.zeros((T,N,D),dtype=np.float32)
	count = 0
	for x,y,t_1 in zip(training_data,Y,t_minus_one_data):
		data3Dtensor[:,count,:] = x
		data3Dtensor_t_1[:,count,:] = t_1
		Y3Dtensor[:,count,:] = y
		count += 1

	# Normalizing the training data features
	data3Dtensor_t_1 = normalizeTensor(data3Dtensor_t_1) #np.divide((data3Dtensor - meanTensor),stdTensor)
	data3Dtensor = normalizeTensor(data3Dtensor) #np.divide((data3Dtensor - meanTensor),stdTensor)
	Y3Dtensor = normalizeTensor(Y3Dtensor) #np.divide((Y3Dtensor - meanTensor),stdTensor)
	return data3Dtensor,Y3Dtensor,data3Dtensor_t_1,N


def addNoise(X_old,X_t_1_old,noise=1e-5):
	X = copy.deepcopy(X_old)
	X_t_1 = copy.deepcopy(X_t_1_old)
	
	nodenames = X.keys()
	[T1,N1,D1] = X[nodenames[0]].shape
	binomial_prob = rng.binomial(1,0.5,size=(T1,N1,1))

	for nm in nodenames:
		noise_to_add = rng.normal(scale=noise,size=X[nm].shape)
		noise_sample = np.repeat(binomial_prob,noise_to_add.shape[2],axis=2) * noise_to_add
		X[nm] += noise_sample
		X_t_1[nm][1:,:,:] += noise_sample[:-1,:,:]
	return X,X_t_1

def addNoiseToFeatures(noise=1e-5):
	global nodeFeatures_noisy,nodeFeatures_t_1_noisy,validate_nodeFeatures_noisy,validate_nodeFeatures_t_1_noisy,forecast_nodeFeatures_noisy,forecast_nodeFeatures_t_1_noisy
	if drop_features:
		[nodeFeatures_noisy,nodeFeatures_t_1_noisy] = addNoise(cherryPickNodeFeatures(randomdropFeaturesfromData(data3Dtensor,drop_id)),nodeFeatures_t_1,noise)
	else:
		[nodeFeatures_noisy,nodeFeatures_t_1_noisy] = addNoise(nodeFeatures,nodeFeatures_t_1,noise)
	[validate_nodeFeatures_noisy,validate_nodeFeatures_t_1_noisy] = addNoise(validate_nodeFeatures,validate_nodeFeatures_t_1,noise)
	[forecast_nodeFeatures_noisy,forecast_nodeFeatures_t_1_noisy] = addNoise(forecast_nodeFeatures,forecast_nodeFeatures_t_1)



def getlabels(nodeName):
	D = predictFeatures[nodeName].shape[2]
	return predictFeatures[nodeName],validate_predictFeatures[nodeName],forecast_predictFeatures[nodeName],forecast_nodeFeatures[nodeName],D

def getfeatures(nodeName,edgeType,nodeConnections,nodeNames,forecast_on_noisy_features=False):
	train_features = getDRAfeatures(nodeName,edgeType,nodeConnections,nodeNames,nodeFeatures_noisy,nodeFeatures_t_1_noisy)
	validate_features = getDRAfeatures(nodeName,edgeType,nodeConnections,nodeNames,validate_nodeFeatures_noisy,validate_nodeFeatures_t_1_noisy)

	forecast_features = []
	if forecast_on_noisy_features:
		forecast_features = getDRAfeatures(nodeName,edgeType,nodeConnections,nodeNames,forecast_nodeFeatures_noisy,forecast_nodeFeatures_t_1_noisy)
	else:
		forecast_features = getDRAfeatures(nodeName,edgeType,nodeConnections,nodeNames,forecast_nodeFeatures,forecast_nodeFeatures_t_1)

	return train_features, validate_features, forecast_features
		
def getDRAfeatures(nodeName,edgeType,nodeConnections,nodeNames,features_to_use,features_to_use_t_1):
	if edgeType.split('_')[1] == 'input':
		if temporal_features:
			return np.concatenate((features_to_use[nodeName],features_to_use[nodeName] - features_to_use_t_1[nodeName]),axis=2)
		else:
			return features_to_use[nodeName]
	
	features = []
	nodesConnectedTo = nodeConnections[nodeName]
	for nm in nodesConnectedTo:
		et1 = nodeNames[nm] + '_' + nodeNames[nodeName]
		et2 = nodeNames[nodeName] + '_' + nodeNames[nm]
		
		f1 = 0
		f2 = 0

		x = 0
		y = 0
		if nm == 'torso':
			x = 0
		if nodeName == 'torso':
			y = 0

		if et1 == et2 and et1 == edgeType:
			f1 = features_to_use[nodeName][:,:,y:] 
			f2 = features_to_use[nm][:,:,x:]
		elif et1 == edgeType:
			f1 = features_to_use[nm][:,:,x:] 
			f2 = features_to_use[nodeName][:,:,y:]
		elif et2 == edgeType:
			f1 = features_to_use[nodeName][:,:,y:] 
			f2 = features_to_use[nm][:,:,x:]
		else:
			continue

		if len(features) == 0:
			features = np.concatenate((f1,f2),axis=2)
		else:
			features += np.concatenate((f1,f2),axis=2)

	return features	


def cherryPickNodeFeatures(data3DTensor):
	Features = {}
	nodeNames = nodeFeaturesRanges.keys()
	for nm in nodeNames:
		filterList = []
		for x in nodeFeaturesRanges[nm]:
			if x not in dimensions_to_ignore:
				filterList.append(x)
		Features[nm] = data3DTensor[:,:,filterList]
	return Features	

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
	N = 4*len(actions)*len(subactions)
	D = trainData[(subject,actions[0],subactions[0])].shape[1]
	trX = np.zeros((prefix,N,D),dtype=np.float32)
	trX_t_1 = np.zeros((prefix,N,D),dtype=np.float32)
	trY = np.zeros((suffix,N,D),dtype=np.float32)
	count = 0
	forecastidx = {}
	for action in actions:
		for i in range(4):
			for subact in subactions:
				data_to_use = []
				if subsample_data:
					data_to_use = trainData[(subject,action,subact,'even')]
				else:
					data_to_use = trainData[(subject,action,subact)]

				T = data_to_use.shape[0]
				idx = rng.randint(16,T-prefix-suffix)
				trX[:,count,:] = data_to_use[idx:(idx+prefix),:]
				trX_t_1[:,count,:] = data_to_use[idx-1:(idx+prefix-1),:]
				trY[:,count,:] = data_to_use[(idx+prefix):(idx+prefix+suffix),:]
				forecastidx[count] = (action,subact,idx)
				count += 1
	toget = num_forecast_examples
	if toget > count:
		toget = count
	return normalizeTensor(trX[:,:toget,:]),normalizeTensor(trX_t_1[:,:toget,:]),normalizeTensor(trY[:,:toget,:]),forecastidx

def addNoiseMalik(X_old,noise=1e-5):
	X = copy.deepcopy(X_old)	
	[T1,N1,D1] = X.shape
	binomial_prob = rng.binomial(1,0.5,size=(T1,N1,1))
	noise_to_add = rng.normal(scale=noise,size=X.shape)
	noise_sample = np.repeat(binomial_prob,noise_to_add.shape[2],axis=2) * noise_to_add
	X += noise_sample
	return X

def getMalikFeatures(noise=1e-5):
	if drop_features:
		return addNoiseMalik(ignoreZeroVarianceFeatures(randomdropFeaturesfromData(data3Dtensor,drop_id))),malikPredictFeatures
	else:
		return addNoiseMalik(malikTrainFeatures,noise=noise),malikPredictFeatures

def getMalikValidationFeatures(noise=1e-5):
	return addNoiseMalik(validate_malikTrainFeatures,noise=noise),validate_malikPredictFeatures

def getMalikTrajectoryForecasting(noise=1e-5):
	return trX_forecast_malik,trY_forecast_malik
	
#Keep T fixed, and tweak delta_shift in order to generate less/more examples
T=150
delta_shift= T - 50
num_forecast_examples = 24
copy_state = 0
full_skeleton = 1
motion_prefix=50
motion_suffix=100
train_for = 'final'
temporal_features = 0
dataset_prefix = ''
crf_file = ''
drop_features = 0
drop_id = [9]

th_len = 3
subsample_data = 1

def runall():
	global trainData,completeData,validateData,completeValidationData,data_stats,data3Dtensor,Y3Dtensor,validate3Dtensor,validateY3Dtensor,trX_forecast,trY_forecast,malikTrainFeatures,malikPredictFeatures,validate_malikTrainFeatures,validate_malikPredictFeatures,trX_forecast_malik,trY_forecast_malik,data_mean,data_std,dimensions_to_ignore,new_idx,nodeFeatures,predictFeatures,validate_nodeFeatures,validate_predictFeatures,forecast_nodeFeatures,forecast_predictFeatures,minibatch_size,forecastidx,drop_id
	global trainSubjects,validateSubject,testSubject,actions,nodeFeatures_t_1,validate_nodeFeatures_t_1,forecast_nodeFeatures_t_1,path_to_dataset,drop_start,drop_end,subsample_data,th_len
	if not subsample_data:
		th_len = 4

	path_to_dataset = '{0}/dataset{1}'.format(base_dir,dataset_prefix)

	if train_for == 'final':
		trainSubjects = ['S1','S6','S7','S8','S9','S11']
		validateSubject = ['S5']
	if train_for == 'final2':
		trainSubjects = ['S1','S6','S7','S8','S9','S5']
		validateSubject = ['S11']
	if train_for == 'smoking':
		trainSubjects = ['S1','S6','S7','S8','S9','S11']
		validateSubject = ['S5']
		actions = ['smoking']
	if train_for == 'eating':
		trainSubjects = ['S1','S6','S7','S8','S9','S11']
		validateSubject = ['S5']
		actions = ['eating']
	if train_for == 'discussion':
		trainSubjects = ['S1','S6','S7','S8','S9','S11']
		validateSubject = ['S5']
		actions = ['discussion']
	if train_for == 'walkingdog':
		trainSubjects = ['S1','S6','S7','S8','S9','S11']
		validateSubject = ['S5']
		actions = ['walkingdog']
#Load training and validation data
	[trainData,completeData]=loadTrainData(trainSubjects)
	[validateData,completeValidationData]=loadTrainData(validateSubject)

#Compute training data mean
	[data_mean,data_std,dimensions_to_ignore,new_idx]=normalizationStats(completeData)
	data_stats = {}
	data_stats['mean'] = data_mean
	data_stats['std'] = data_std
	data_stats['ignore_dimensions'] = dimensions_to_ignore
	print T
#Create normalized 3D tensor for training and validation

	if copy_state:
		[data3Dtensor,Y3Dtensor,minibatch_size] = sampleConnectedTrainSequences(trainData,T,delta_shift)
		[validate3Dtensor,validateY3Dtensor,minibatch_size_ignore] = sampleConnectedTrainSequences(validateData,T,delta_shift)
	else:
		[data3Dtensor,Y3Dtensor,data3Dtensor_t_1,minibatch_size] = sampleTrainSequences(trainData,T,delta_shift)
		[validate3Dtensor,validateY3Dtensor,validate3Dtensor_t_1,minibatch_size_ignore] = sampleTrainSequences(validateData,T,delta_shift)

	print 'Training data stats (T,N,D) is ',data3Dtensor.shape
	print 'Training data stats (T,N,D) is ',validate3Dtensor.shape

	if drop_features:
		[validate3Dtensor,drop_start,drop_end] = dropFeaturesfromData(validate3Dtensor,drop_id)


#Generate normalized data for trajectory forecasting
	trX_forecast,trX_forecast_t_1,trY_forecast,forecastidx = generateForecastingExamples(validateData,motion_prefix,motion_suffix,validateSubject[0])

	data_stats['forecastidx'] = forecastidx
#Create training and validation features for DRA
	nodeFeatures = cherryPickNodeFeatures(data3Dtensor)
	nodeFeatures_t_1 = cherryPickNodeFeatures(data3Dtensor_t_1)
	validate_nodeFeatures = cherryPickNodeFeatures(validate3Dtensor)
	validate_nodeFeatures_t_1 = cherryPickNodeFeatures(validate3Dtensor_t_1)
	forecast_nodeFeatures = cherryPickNodeFeatures(trX_forecast)
	forecast_nodeFeatures_t_1 = cherryPickNodeFeatures(trX_forecast_t_1)

	predictFeatures = cherryPickNodeFeatures(Y3Dtensor)
	validate_predictFeatures = cherryPickNodeFeatures(validateY3Dtensor)
	forecast_predictFeatures = cherryPickNodeFeatures(trY_forecast)

#Create training and validation features for Malik's LSTM model
	malikTrainFeatures = ignoreZeroVarianceFeatures(data3Dtensor)
	malikPredictFeatures = ignoreZeroVarianceFeatures(Y3Dtensor)
	validate_malikTrainFeatures = ignoreZeroVarianceFeatures(validate3Dtensor)
	validate_malikPredictFeatures = ignoreZeroVarianceFeatures(validateY3Dtensor)
	trX_forecast_malik = ignoreZeroVarianceFeatures(trX_forecast)
	trY_forecast_malik = ignoreZeroVarianceFeatures(trY_forecast)

def randomdropFeaturesfromData(datatensor,drop_id):
	print 'Dropping features from training set'
	dataTensor = copy.deepcopy(datatensor)
	[T,N,D] = dataTensor.shape
	drop_joints = rng.binomial(1,0.4,size=(T,N,1))
	dataTensor[:,:,drop_id] = np.repeat(drop_joints,len(drop_id),axis=2) * dataTensor[:,:,drop_id]
	return dataTensor


def dropFeaturesfromData(dataTensor,drop_id):
	print 'Dropping features from validation set'
	T = dataTensor.shape[0]
	start_idx = rng.randint(17,T-11)
	end_idx = start_idx + 10
	dataTensor[start_idx:end_idx,:,drop_id] = np.float32(0.0)
	return dataTensor,start_idx,end_idx

