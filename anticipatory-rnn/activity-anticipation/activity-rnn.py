import numpy as np
import theano
from theano import tensor as T
from readData import sortActivities
from neuralmodels.utils import permute, load
from neuralmodels.costs import softmax_loss
from neuralmodels.models import RNN
from neuralmodels.predictions import OutputMaxProb, OutputSampleFromDiscrete
from neuralmodels.layers import softmax, simpleRNN, OneHot, LSTM, TemporalInputFeatures
import cPickle

def text_prediction(class_ids_reverse,p_labels):
	N = p_labels.shape[1]
	T = p_labels.shape[0]
	text_output = []
	for i in range(N):
		t = ''
		for j in p_labels[:,i]:
			t = t + class_ids_reverse[j]
		text_output.append(t)
	return text_output

if __name__ == '__main__':
	
	test_data = cPickle.load(open('dataset/test_data_4.pik'))	
	Y_te = test_data['labels']
	X_te = test_data['features']

	#print X_te.shape
	#print Y_te.shape

	train_data = cPickle.load(open('dataset/train_data_4.pik'))	
	Y_tr = train_data['labels']
	X_tr = train_data['features']
	print X_tr.shape
	print Y_tr.shape

	

	num_train = X_tr.shape[1]
	num_test = len(X_te)
	len_samples = X_tr.shape[0]

	num_classes = int(np.max(Y_tr) - np.min(Y_tr) + 1)
	inputD = X_tr.shape[2]
	outputD = num_classes 

	print 'Number of classes ',outputD
	print 'Feature dimension ',inputD

	epochs = 80
	batch_size = num_train
	learning_rate_decay = 0.97
	decay_after = 5
	
	use_pretrained = False

	global rnn
	if not use_pretrained:
		# Creating network layers
		layers = [TemporalInputFeatures(inputD),LSTM('tanh','sigmoid','orthogonal',4),softmax(num_classes)]

		trY = T.lmatrix()

		# Initializing network
		rnn = RNN(layers,softmax_loss,trY,1e-3)

		# Fitting model
		rnn.fitModel(X_tr,Y_tr,1,'checkpoints/',epochs,batch_size,learning_rate_decay,decay_after)
	else:
		# Prediction
		rnn = load('checkpoints/checkpoint.69')

	predictions = []
	errors = 0
	N = 0
	for xte,yte in zip(X_te,Y_te):
		prediction = rnn.predict_sequence(xte,yte,OutputMaxProb)
		predictions.append(prediction)
		t = np.nonzero(yte-prediction)
		print t
		errors += len(t[0])
		N += yte.shape[0]
	cPickle.dump(predictions,open('predict.pik','wb'))
	print 'error = {0}'.format(errors*1.0/N)
	#cPickle.dump(Y_te,open('test.pik','wb'))

	'''
	folder = '/home/ashesh/Downloads/features_cad120_ground_truth_segmentation/features_binary_svm_format'
	[Y,X] = sortActivities(folder)

	num_samples = X.shape[1]
	num_validation = int(0.2*num_samples)
	num_train = num_samples - num_validation
	len_samples = X.shape[0]

	epochs = 50
	batch_size = num_train
	learning_rate_decay = 0.97
	decay_after=5

	num_classes = int(np.max(Y) - np.min(Y) + 1)
	inputD = X.shape[2]
	outputD = num_classes 

	permutation = permute(num_samples)
	X = X[:,permutation,:]
	Y = Y[:,permutation]
	X_tr = X[:,:num_train,:]
	Y_tr = Y[:,:num_train]
	X_valid = X[:,num_train:,:]
	Y_valid = Y[:,num_train:]

	print X.shape
	print Y.shape

	print X_tr.shape
	print Y_tr.shape

	rnn = load('checkpoints/checkpoint.49')

	prediction = rnn.predict_sequence(X_valid,Y_valid,OutputMaxProb)
	print prediction.shape
	print Y_valid.shape
	cPickle.dump(prediction,open('p.pik','wb'))
	cPickle.dump(Y_valid,open('t.pik','wb'))
	# Creating network layers
	layers = [TemporalInputFeatures(inputD),LSTM(),softmax(num_classes)]

	trY = T.lmatrix()

	# Initializing network
	rnn = RNN(layers,softmax_loss,trY,1e-3)

	# Fitting model
	rnn.fitModel(X_tr,Y_tr,1,'checkpoints/',epochs,batch_size,learning_rate_decay,decay_after)
	'''	
