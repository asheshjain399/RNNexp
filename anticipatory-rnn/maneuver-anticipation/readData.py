import os
import numpy as np
import csv
from os import listdir
import random
from neuralmodels.dataAugmentation import sampleSubSequences
from utils import sixDigitRandomNum
import cPickle

def readFeatures(fname):
	f = open(fname,'rb')
	rows = csv.reader(f,delimiter=',')
	data = []
	for row in rows:
		data.append([float(r) for r in row])
	data = np.array(data)
	f.close()
	return data.T
	# dim = T x D

def iterateThroughFiles(folder):
	list_of_files = listdir(folder)

	features = []
	for files in list_of_files:
		features.append(readFeatures(folder+'/'+files))
	return features

def readManeuvers(folder):
	features = {}
	class_wise_count = {}
	sample_ratio = {}
	N = 0.0
	for action in actions:
		features[action] = iterateThroughFiles(folder+'/'+action)
		class_wise_count[action] = 1.0*len(features[action])
		N = N + class_wise_count[action]
	for action in actions:
		if use_sample_ratio:
			#sample_ratio[action] = 1.0 - ( class_wise_count[action]/N )
			sample_ratio[action] = class_wise_count['end_action']/class_wise_count[action]
		else:
			sample_ratio[action] = 1.0

	return features, sample_ratio

def createData(folder):
	features_train,sample_train_ratio = readManeuvers(folder+'/train')
	if use_data_augmentation:
		[N_train,features_train] = multiplyData(features_train,sample_train_ratio)
	[N_train,Tmax,Y_train,features_train] = createLabels(features_train)
	[Y_train,features_train] = processFeatures(Y_train,features_train,Tmax,N_train)	


	features_test,_ = readManeuvers(folder+'/test')
	[N_test,Tmax,Y_test,features_test] = createLabels(features_test)
	[Y_test,features_test] = reshapeData(Y_test,features_test)

	train_data = {'params':params,'labels':Y_train,'features':features_train,'actions':actions}
	test_data = {'labels':Y_test,'features':features_test,'actions':actions}


	prefix = sixDigitRandomNum()	
	cPickle.dump(train_data,open('dataset/train_data_{0}.pik'.format(prefix),'wb'))
	cPickle.dump(test_data,open('dataset/test_data_{0}.pik'.format(prefix),'wb'))

	print 'T={0} N={1}'.format(Tmax,N_train)
	print 'Saving prefix as {0}'.format(prefix)

def processFeatures(y,node,T,N):
	D = node[0].shape[1]
	features = np.zeros((T,N,D))
	Y = np.zeros((T,N),dtype=np.int64)

	count = 0
	for l, n in zip(y,node):
		assert(n.shape[1] == D)

		t = n.shape[0]
		n = np.reshape(n, (n.shape[0],1,n.shape[1]))
		features[T-t:,count:count+1,:] = n
		
		Y[T-t:,count] = l

		count += 1
	return Y,features


def createLabels(features):
	X = []
	Y = []
	N = 0
	Tmax = 0
	for action in actions:
		for f in features[action]:
			X.append(f)
			T = f.shape[0]
			if T > Tmax:
				Tmax = T
			Y.append(np.array([1+actions.index(action)]*T))
			N += 1
	return N,Tmax,Y,X

def reshapeData(y,node):
	y_ = []
	features = []
	
	for l,n in zip(y,node):
		y_.append(np.reshape(l,(l.shape[0],1)))
		temp = np.zeros((n.shape[0],1,n.shape[1]))
		temp[:,0,:] = n
		features.append(temp)
	return y_,features

def multiplyData(features,sample_ratio):
	N = 0	
	for action in actions:
		new_samples = []
		for f in features[action]:
			N += 1
			samples = sampleSubSequences(f.shape[0],int(sample_ratio[action]*extra_samples),min_length_sequence)
			for s in samples:
				N += 1
				if copy_start_state:			
					ll = [0]
					if s[0] > 0:
						ll = ll + range(s[0],s[1])
					else:
						ll = range(s[0],s[1])
					new_samples.append(f[ll,:])
				else:
					new_samples.append(f[s[0]:s[1],:])
		features[action] = features[action] + new_samples
		print '{0} {1}'.format(action,len(features[action]))
	return N,features

if __name__=='__main__':
	global min_length_sequence, use_data_augmentation, extra_samples, copy_start_state, params, actions, use_sample_ratio
	use_data_augmentation = True
	min_length_sequence = 4
	extra_samples = 5
	copy_start_state = True
	use_sample_ratio = True
	params = {
		'use_data_augmentation':use_data_augmentation,
		'min_length_sequence':min_length_sequence,
		'extra_samples':extra_samples,
		'copy_start_state':copy_start_state,
		}
	actions = ['end_action','lchange','rchange','lturn','rturn']
	folder = '/home/ashesh/project/Brain4Cars/Software/HMMBaseline/observations/all/AIOHMM_I_O/fold_1'
	createData(folder)
