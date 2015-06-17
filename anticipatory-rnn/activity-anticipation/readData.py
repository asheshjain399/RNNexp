import numpy as np
import re
from os import listdir
import random
from neuralmodels.dataAugmentation import sampleSubSequences
from utils import sixDigitRandomNum
import cPickle



def readFeatures(ll):
	colon_seperated = [x.strip() for x in ll.strip().spilt(' ')]
	f_list = [int(x.split(':')[1]) for x in colon_seperated]
	return f_list

def parseColonSeperatedFeatures(colon_seperated):
	f_list = [int(x.split(':')[1]) for x in colon_seperated]
	return f_list

def parseSegment(folder,filename):
	f = open(folder + '/' + filename,'r')
	lines = f.readlines()
	f.close()
	
	node_stats = [int(x) for x in lines[0].strip().split(' ')]
	num_o = node_stats[0]
	num_o_o_e = node_stats[1]
	num_s_o_e = node_stats[2]
	num_affordances = node_stats[3]
	num_sub_activities = node_stats[4]


	o_aff = []
	o_id = []
	o_fea = []
	for l in lines[1:(num_o+1)]:
		splitted_str = l.strip().split(' ')
		o_aff.append(int(splitted_str[0]))
		o_id.append(int(splitted_str[1]))
		o_fea.append(parseColonSeperatedFeatures(splitted_str[2:]))
	
	skeleton_stats = lines[num_o+1].strip().split(' ')
	sub_activity = int(skeleton_stats[0])
	s_features = parseColonSeperatedFeatures(skeleton_stats[2:]) 

	o_o_id = []
	o_o_fea = []
	for l in lines[num_o+2:num_o+2+num_o_o_e]:
		splitted_str = l.strip().split(' ')
		o_o_id.append([int(splitted_str[2]),int(splitted_str[3])])
		o_o_fea.append(parseColonSeperatedFeatures(splitted_str[4:]))

	s_o_id = []
	s_o_fea = []
	for l in lines[num_o+2+num_o_o_e : num_o+2+num_o_o_e+num_s_o_e]:
		splitted_str = l.strip().split(' ')
		s_o_id.append(int(splitted_str[2]))
		s_o_fea.append(parseColonSeperatedFeatures(splitted_str[3:]))
	
	return {
		'o_aff':o_aff,
		'o_id':o_id,
		'o_fea':o_fea,
		'o_o_id':o_o_id,
		'o_o_fea':o_o_fea,
		's_o_id':s_o_id,
		's_o_fea':s_o_fea,
		'sub_activity':sub_activity,
		'sub_activity_features':s_features
		}

def parseTemporalEdge(folder,filename):
	f = open(folder + '/' + filename,'r')
	lines = f.readlines()
	f.close()

	node_stats = [int(x) for x in lines[0].strip().split(' ')]
	num_o_o_e = node_stats[0]

	o_id = []
	o_o_fea = []
	for l in lines[1:(num_o_o_e+1)]:
		splitted_str = l.strip().split(' ')
		o_id.append(int(splitted_str[2]))
		o_o_fea.append(parseColonSeperatedFeatures(splitted_str[3:]))
	
	skeleton_stats = lines[num_o_o_e+1].strip().split(' ')
	s_s_features = parseColonSeperatedFeatures(skeleton_stats[3:]) 

	return {
		'o_id':o_id,
		'o_o_fea':o_o_fea,
		's_s_features':s_s_features
		}	


def readActivity(folder,files):	
	input_features_node = []
	input_features_edge = []
	Y = []
	for f in files:
		if len(f.split('_')) == 2:
			key_value = parseSegment(folder,f)
			input_features_node.append(key_value['sub_activity_features'])
			Y.append(key_value['sub_activity'])
		elif len(f.split('_')) == 3:
			key_value = parseTemporalEdge(folder,f)
			input_features_edge.append(key_value['s_s_features'])
	#print Y
	input_features_edge.insert(0,[0]*len(input_features_edge[0]))
	#input_features_edge = [ [0]*len(input_features_edge[0]), input_features_edge]

	# only subactivity features
	#print files
	Y = np.array(Y)
	input_features_node = np.array(input_features_node)
	input_features_edge = np.array(input_features_edge)

	if not (input_features_node.shape[0] == input_features_edge.shape[0]):
		input_features_edge = input_features_edge[:-1]
	assert(input_features_node.shape[0] == input_features_edge.shape[0])

	return Y, input_features_node, input_features_edge

def sortActivities(folder):
	T = 0
	N = 0
	D_node = 0
	D_edge = 0
	Y = []
	features = []

	all_the_files = listdir(folder)

	all_activities = []
	activities_time_steps = {}
	for f in all_the_files:
		s = f.split('_')[0]
		if s not in all_activities:
			all_activities.append(s)
			activities_time_steps[s] = 1
		else:
			activities_time_steps[s] += 1
	
	T = int((max(activities_time_steps.values())+1)/2)
	N = len(all_activities)
	print 'max time ',T


	for i in range(5):
		random.shuffle(all_activities)
	
	N_train = int(0.8*N)
	N_test = N - N_train


	[y,node,edge,D_node,D_edge] = appendFeatures(folder,all_activities[:N_train])

	if use_data_augmentation:
		[N_train_multiply,y,node,edge] = multiplyData(y,node,edge)

	[Y_train,features_train] = processFeatures(y,node,edge,T,N_train_multiply,D_node,D_edge)

	[y,node,edge,D_node,D_edge] = appendFeatures(folder,all_activities[N_train:])
	#[Y_test,features_test] = processFeatures(y,node,edge,T,N_test,D_node,D_edge)
	[Y_test,features_test] = reshapeData(y,node,edge,D_node,D_edge)

	train_data = {'params':params,'labels':Y_train,'features':features_train}
	test_data = {'labels':Y_test,'features':features_test}


	prefix = sixDigitRandomNum()	
	cPickle.dump(train_data,open('dataset/train_data_{0}.pik'.format(prefix),'wb'))
	cPickle.dump(test_data,open('dataset/test_data_{0}.pik'.format(prefix),'wb'))

	print 'T={0} N={1} D={2}'.format(T,N,(D_node+D_edge))
	print 'Saving prefix as {0}'.format(prefix)
	#return Y,features

def reshapeData(y,node,edge,D_node,D_edge):
	y_ = []
	features = []

	for l,n,e in zip(y,node,edge):
		y_.append(np.reshape(l,(l.shape[0],1)))
		temp = np.zeros((n.shape[0],1,D_node+D_edge))
		temp[:,0,:D_node] = n
		temp[:,0,D_node:] = e
		features.append(temp)
	return y_,features

def appendFeatures(folder,all_activities):
	all_the_files = listdir(folder)
	y=[]
	node=[]
	edge=[]
	D_node = 0
	D_edge = 0
	for activity in all_activities:
		filenames = []
		idx = 1

		# Gathering all the files for a given activity
		while(True):
			f = '{0}_{1}.txt'.format(activity,idx)			
			if f not in all_the_files:
				break
			filenames.append(f)

			f = '{0}_{1}_{2}.txt'.format(activity,idx,idx+1)			
			if f not in all_the_files:
				break
			filenames.append(f)

			idx += 1
		y_,node_,edge_ = readActivity(folder,filenames)
			
		D_node = node_.shape[1]
		D_edge = edge_.shape[1]
		y.append(y_)
		node.append(node_)
		edge.append(edge_)
	return y,node,edge,D_node,D_edge

def processFeatures(y,node,edge,T,N,D_node,D_edge):
	D = D_node + D_edge
	features = np.zeros((T,N,D))
	Y = np.zeros((T,N),dtype=np.int64)

	count = 0
	for l, n , e in zip(y,node,edge):
		assert(n.shape[0] == e.shape[0])
		assert(n.shape[1] == D_node)
		assert(e.shape[1] == D_edge)

		t = n.shape[0]
		n = np.reshape(n, (n.shape[0],1,n.shape[1]))
		features[T-t:,count:count+1,:D_node] = n

		e = np.reshape(e, (e.shape[0],1,e.shape[1]))
		features[T-t:,count:count+1,D_node:] = e
		
		Y[T-t:,count] = l

		count += 1
	return Y,features

def multiplyData(y,node,edge):
	y_ = []
	node_ = []
	edge_ = []
	N = len(node)
	for l, n , e in zip(y,node,edge):
		samples = sampleSubSequences(n.shape[0],extra_samples,min_length_sequence)
		for s in samples:
			if copy_start_state:			
				ll = [0]
				if s[0] > 0:
					ll = ll + range(s[0],s[1])
				else:
					ll = range(s[0],s[1])
				y_.append(l[ll])
				node_.append(n[ll,:])
				edge_.append(e[ll,:])
			else:
				y_.append(l[s[0]:s[1]])
				node_.append(n[s[0]:s[1],:])
				edge_.append(e[s[0]:s[1],:])
			N += 1
	y = y + y_
	edge = edge + edge_
	node = node + node_
	return N,y,node,edge

if __name__ == '__main__':

	global min_length_sequence, use_data_augmentation, extra_samples, copy_start_state, params
	use_data_augmentation = True
	min_length_sequence = 4
	extra_samples = 100
	copy_start_state = True
	params = {
		'use_data_augmentation':use_data_augmentation,
		'min_length_sequence':min_length_sequence,
		'extra_samples':extra_samples,
		'copy_start_state':copy_start_state,
		}
	s='/home/ashesh/Downloads/features_cad120_ground_truth_segmentation/features_binary_svm_format'
	sortActivities(s)
