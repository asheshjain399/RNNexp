import numpy as np
import re
from os import listdir
import random
from neuralmodels.dataAugmentation import sampleSubSequences
from utils import sixDigitRandomNum
import cPickle
import sys
import os
import pdb

def readFeatures(ll):
	colon_seperated = [x.strip() for x in ll.strip().spilt(' ')]
	f_list = [int(x.split(':')[1]) for x in colon_seperated]
	return f_list

def parseColonSeperatedFeatures(colon_seperated):
	f_list = [int(x.split(':')[1]) for x in colon_seperated]
	return f_list

def activityStats(folder,filename):
	f = open(folder + '/' + filename,'r')
	lines = f.readlines()
	f.close()
	
	node_stats = [int(x) for x in lines[0].strip().split(' ')]
	num_o = node_stats[0]
	num_o_o_e = node_stats[1]
	num_s_o_e = node_stats[2]
	num_affordances = node_stats[3]
	num_sub_activities = node_stats[4]
	
	return num_o,num_o_o_e,num_s_o_e

def parseSegment(folder,filename):
	global D_node_human # = 0 # Node feature length
	global D_node_object # = 0 # Node feature length
	global D_human_object # = 0 # Intra human-object edge feature length
	global D_object_object # = 0 # Intra object-object edge feature length 

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
		D_node_object = len(o_fea[-1])
	
	skeleton_stats = lines[num_o+1].strip().split(' ')
	sub_activity = int(skeleton_stats[0])
	s_features = parseColonSeperatedFeatures(skeleton_stats[2:]) 
	D_node_human = len(s_features)

	o_o_id = []
	o_o_fea = []
	for l in lines[num_o+2:num_o+2+num_o_o_e]:
		splitted_str = l.strip().split(' ')
		o_o_id.append([int(splitted_str[2]),int(splitted_str[3])])
		o_o_fea.append(parseColonSeperatedFeatures(splitted_str[4:]))
		D_object_object = int(2*len(o_o_fea[-1]))

	s_o_id = []
	s_o_fea = []
	for l in lines[num_o+2+num_o_o_e : num_o+2+num_o_o_e+num_s_o_e]:
		splitted_str = l.strip().split(' ')
		s_o_id.append(int(splitted_str[2]))
		s_o_fea.append(parseColonSeperatedFeatures(splitted_str[3:]))
		D_human_object = len(s_o_fea[-1])
		
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
	global D_edge_human # = 0 # Temporal human edge feature length
	global D_edge_object # = 0 # Temporal object edge feature length
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
		D_edge_object = len(o_o_fea[-1])	

	skeleton_stats = lines[num_o_o_e+1].strip().split(' ')
	s_s_features = parseColonSeperatedFeatures(skeleton_stats[3:]) 
	D_edge_human = len(s_s_features)

	return {
		'o_id':o_id,
		'o_o_fea':o_o_fea,
		's_s_features':s_s_features
		}	


def readActivity(folder,files):	

	features_node = {}
	features_node_node = {}
	features_temporal_edge = {}
	Y = {}

	num_o,num_o_o_e,num_s_o_e = activityStats(folder,files[0])

	Y['h'] = []
	features_node['h'] = []
	features_temporal_edge['h'] = []
	features_node_node['h'] = {}
	for i in range(1,num_o+1):
		features_node[str(i)] = []
		features_temporal_edge[str(i)] = []
		features_node_node['h'][str(i)] = []
		features_node_node[str(i)] = {}
		Y[str(i)] = []

		for j in range(1,num_o+1):
			if i == j:
				continue
			features_node_node[str(i)][str(j)] = []
	
	for f in files:
		if len(f.split('_')) == 2:
			key_value = parseSegment(folder,f)
			features_node['h'].append(key_value['sub_activity_features'])
			Y['h'].append(key_value['sub_activity'])
			for i in range(len(key_value['o_id'])):
				o_id = key_value['o_id'][i]
				o_fea = key_value['o_fea'][i]
				o_aff = key_value['o_aff'][i]
				features_node[str(o_id)].append(o_fea)
				Y[str(o_id)].append(o_aff)

			for i in range(len(key_value['s_o_id'])):
				s_o_id = key_value['s_o_id'][i]
				s_o_fea = key_value['s_o_fea'][i]
				features_node_node['h'][str(s_o_id)].append(s_o_fea)

			for i in range(len(key_value['o_o_id'])):
				o_o_id = key_value['o_o_id'][i]
				o_o_fea = key_value['o_o_fea'][i]
				features_node_node[str(o_o_id[0])][str(o_o_id[1])].append(o_o_fea)

		elif len(f.split('_')) == 3:
			key_value = parseTemporalEdge(folder,f)
			features_temporal_edge['h'].append(key_value['s_s_features'])

			for i in range(len(key_value['o_id'])):
				o_id = key_value['o_id'][i]
				o_o_fea = key_value['o_o_fea'][i]
				features_temporal_edge[str(o_id)].append(o_o_fea)


	for k in features_temporal_edge.keys():
		features_temporal_edge[k].insert(0,[0]*len(features_temporal_edge[k][0]))
	
	for k in Y.keys():
		Y[k] = np.array(Y[k])
	
	for k in features_node:
		features_node[k] = np.array(features_node[k])

	for k in features_temporal_edge.keys():
		features_temporal_edge[k] = np.array(features_temporal_edge[k])

		if not (features_node['h'].shape[0] == features_temporal_edge[k].shape[0]):
			features_temporal_edge[k] = features_temporal_edge[k][:-1]

	for k in features_node_node.keys():
		for k2 in features_node_node[k].keys():
			features_node_node[k][k2] = np.array(features_node_node[k][k2])

	for k in features_temporal_edge.keys():
		assert(features_node['h'].shape[0] == features_temporal_edge[k].shape[0])

	return Y, features_node, features_temporal_edge, features_node_node

def sortActivities(folder):
	ground_truth='/scail/scratch/group/cvgl/ashesh/activity-anticipation/features_ground_truth'
	dataset = '/scail/scratch/group/cvgl/ashesh/activity-anticipation/dataset/fold_{0}'.format(fold)
	if not os.path.exists(dataset):
		os.mkdir(dataset)

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

	'''
	for i in range(5):
		random.shuffle(all_activities)
	
	N_train = int(0.8*N)
	N_test = N - N_train
	'''
	N_train = len(train_activities)
	N_test = len(test_activities)

	[y,node,edge,edge_intra,y_object,node_object,edge_object,edge_intra_object,edge_intra_object_human] = appendFeatures(folder,train_activities)
	y_anticipation = [] 
	y_object_anticipation = []
	if use_data_augmentation:
		[N_train_multiply,y,node,edge,edge_intra,y_object,node_object,edge_object,edge_intra_object,edge_intra_object_human,y_anticipation,y_object_anticipation] = multiplyData(y,node,edge,edge_intra,y_object,node_object,edge_object,edge_intra_object,edge_intra_object_human)
	[Y_human,Y_human_anticipation,features_human_disjoint,features_human_shared] = processFeatures(y,y_anticipation,[node,edge],[edge_intra],[D_node_human,D_edge_human],[D_human_object],T)
	[Y_objects,Y_objects_anticipation,features_objects_disjoint,features_objects_shared] = processFeatures(y_object,y_object_anticipation,[node_object,edge_object,edge_intra_object],[edge_intra_object_human],[D_node_object,D_edge_object,D_object_object],[D_human_object],T)
	print "N_human = ",Y_human.shape[1]
	print "N_object = ",Y_objects.shape[1]
	train_data = {'params':params,'labels_human':Y_human,'labels_objects':Y_objects,'labels_human_anticipation':Y_human_anticipation,'labels_objects_anticipation':Y_objects_anticipation,'features_human_disjoint':features_human_disjoint,'features_human_shared':features_human_shared,'features_objects_disjoint':features_objects_disjoint,'features_objects_shared':features_objects_shared}
	cPickle.dump(train_data,open('{1}/train_data_{0}.pik'.format(prefix,dataset),'wb'))

	[y,node,edge,edge_intra,y_object,node_object,edge_object,edge_intra_object,edge_intra_object_human] = appendFeatures(folder,test_activities)
	[Y_human,Y_human_anticipation,features_human_disjoint,features_human_shared] = reshapeData(y,[node,edge],[edge_intra],[D_node_human,D_edge_human],[D_human_object],11)
	y_object,node_object,edge_object,edge_intra_object,edge_intra_object_human = deserializedata(y_object,node_object,edge_object,edge_intra_object,edge_intra_object_human)	
	[Y_objects,Y_objects_anticipation,features_objects_disjoint,features_objects_shared] = reshapeData(y_object,[node_object,edge_object,edge_intra_object],[edge_intra_object_human],[D_node_object,D_edge_object,D_object_object],[D_human_object],13)
	test_data = {'params':params,'labels_human':Y_human,'labels_objects':Y_objects,'labels_human_anticipation':Y_human_anticipation,'labels_objects_anticipation':Y_objects_anticipation,'features_human_disjoint':features_human_disjoint,'features_human_shared':features_human_shared,'features_objects_disjoint':features_objects_disjoint,'features_objects_shared':features_objects_shared}
	cPickle.dump(test_data,open('{1}/test_data_{0}.pik'.format(prefix,dataset),'wb'))

	[y,node,edge,edge_intra,y_object,node_object,edge_object,edge_intra_object,edge_intra_object_human] = appendFeatures(ground_truth,test_activities)
	[Y_human,Y_human_anticipation,features_human_disjoint,features_human_shared] = reshapeData(y,[node,edge],[edge_intra],[D_node_human,D_edge_human],[D_human_object],11)
	y_object,node_object,edge_object,edge_intra_object,edge_intra_object_human = deserializedata(y_object,node_object,edge_object,edge_intra_object,edge_intra_object_human)	
	[Y_objects,Y_objects_anticipation,features_objects_disjoint,features_objects_shared] = reshapeData(y_object,[node_object,edge_object,edge_intra_object],[edge_intra_object_human],[D_node_object,D_edge_object,D_object_object],[D_human_object],13)
	test_data = {'params':params,'labels_human':Y_human,'labels_objects':Y_objects,'labels_human_anticipation':Y_human_anticipation,'labels_objects_anticipation':Y_objects_anticipation,'features_human_disjoint':features_human_disjoint,'features_human_shared':features_human_shared,'features_objects_disjoint':features_objects_disjoint,'features_objects_shared':features_objects_shared}
	cPickle.dump(test_data,open('{1}/grount_truth_test_data_{0}.pik'.format(prefix,dataset),'wb'))

	print 'Saving prefix as {0}'.format(prefix)
	#pdb.set_trace()
	#return Y,features

def reshapeData(y,node_disjoint,node_shared,D_disjoint_list,D_shared_list,label_append=11):
	D_disjoint = 0
	for D in D_disjoint_list:
		D_disjoint += D

	D_shared = 0
	for D in D_shared_list:
		D_shared += D

	y_ = []
	y_anticipation_ = []
	features_disjoint = []
	features_shared = []
	N = len(y)
	
	for i in range(N):
		t = y[i].shape[0]
		y_.append(np.reshape(y[i],(t,1)))

		y_arr = y[i]
		y_temp = appendToArray(y_arr[1:],label_append)
		y_anticipation_.append(np.reshape(y_temp,(t,1)))

		temp = np.zeros((t,1,D_disjoint),dtype=np.float32)
		d_start = 0
		for nd, dd in zip(node_disjoint,D_disjoint_list):
			temp[:,0,d_start:d_start+dd] = nd[i]
			d_start += dd
		features_disjoint.append(temp)

		temp = np.zeros((t,1,D_shared),dtype=np.float32)
		d_start = 0
		for ns, ds in zip(node_shared,D_shared_list):
			temp[:,0,d_start:d_start+ds] = ns[i]
			d_start += ds
		features_shared.append(temp)

	return y_,y_anticipation_,features_disjoint,features_shared
	
def appendFeatures(folder,all_activities):
	all_the_files = listdir(folder)
	y_human = []
	node_human = []
	temporal_edge_human = []
	intra_human_object = []

	# For every object node
	y_object_ = []
	node_object_ = []
	temporal_edge_object_ = []
	intra_object_object_ = []
	intra_object_human_ = []
	
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
		readActivity(folder,filenames)

	for activity in all_activities:
		filenames = []
		idx = 1
		y_object = []
		node_object = []
		temporal_edge_object = []
		intra_object_object = []
		intra_object_human = []

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
		y_,node_,temporal_edge_,intra_edge_ = readActivity(folder,filenames)
			
		y_human.append(y_['h'])
		node_human.append(node_['h'])
		temporal_edge_human.append(temporal_edge_['h'])
		intra_h_o = intra_edge_['h']['1']
		for k in intra_edge_['h'].keys():
			if k == '1':
				continue
			intra_h_o += intra_edge_['h'][k]
		#intra_human_object.append(((1.0/len(intra_edge_['h'].keys()))*intra_h_o))
		intra_human_object.append(intra_h_o)

		object_ids = y_.keys()
		del object_ids[object_ids.index('h')]
		for oid in object_ids:
			y_object.append(y_[oid])
			node_object.append(node_[oid])
			temporal_edge_object.append(temporal_edge_[oid])
			intra_object_human.append(intra_edge_['h'][oid])
			intra_o_o = np.zeros((node_[oid].shape[0],D_object_object))
			for _oid in object_ids:
				if _oid == oid:
					continue
				intra_o_o[:,:intra_edge_[oid][_oid].shape[1]] += intra_edge_[oid][_oid]
				intra_o_o[:,intra_edge_[oid][_oid].shape[1]:] += intra_edge_[_oid][oid]
			intra_object_object.append(intra_o_o)
		y_object_.append(y_object)
		node_object_.append(node_object)
		temporal_edge_object_.append(temporal_edge_object)
		intra_object_object_.append(intra_object_object)
		intra_object_human_.append(intra_object_human)
	
	return y_human, node_human, temporal_edge_human, intra_human_object, y_object_, node_object_, temporal_edge_object_, intra_object_object_, intra_object_human_

def processFeatures(y,y_anticipation,node_disjoint,node_shared,D_disjoint_list,D_shared_list,T):
	N = len(y)
	assert(N == len(y_anticipation))
	D_disjoint = 0
	for D in D_disjoint_list:
		D_disjoint += D

	D_shared = 0
	for D in D_shared_list:
		D_shared += D
	
	features_disjoint = np.zeros((T,N,D_disjoint),dtype=np.float32)
 	features_shared = np.zeros((T,N,D_shared),dtype=np.float32)
	Y = np.zeros((T,N),dtype=np.int64)
	Y_anticipation = np.zeros((T,N),dtype=np.int64)

	for i in range(N):
		d_start = 0
		t = y[i].shape[0]
		for nd, dd in zip(node_disjoint,D_disjoint_list):
			node = nd[i]
			assert(t == node.shape[0])
			assert(dd == node.shape[1])
			features_disjoint[T-t:,i:i+1,d_start:d_start+dd] = np.reshape(node,(t,1,dd))
			d_start += dd
	
		d_start = 0
		for ns, ds in zip(node_shared,D_shared_list):
			node = ns[i]
			assert(t == node.shape[0])
			assert(ds == node.shape[1])
			features_shared[T-t:,i:i+1,d_start:d_start+ds] = np.reshape(node,(t,1,ds))
			d_start += ds
		Y[T-t:,i] = y[i]
		Y_anticipation[T-t:,i] = y_anticipation[i]

	return Y,Y_anticipation,features_disjoint,features_shared

def deserializedata(y_object,node_object,edge_object,edge_intra_object,edge_intra_object_human):
	y_object_ = []
	node_object_ = []
	edge_object_ = []
	edge_intra_object_ = []
	edge_intra_object_human_ = []
	for yo, no, eo, eio, eioh in zip(y_object,node_object,edge_object,edge_intra_object,edge_intra_object_human):
		for yo_, no_, eo_, eio_, eioh_ in zip(yo, no, eo, eio, eioh):
			y_object_.append(yo_)
			node_object_.append(no_)
			edge_object_.append(eo_)
			edge_intra_object_.append(eio_)
			edge_intra_object_human_.append(eioh_)
	return y_object_,node_object_,edge_object_,edge_intra_object_,edge_intra_object_human_

def appendToArray(a,add,choose_list=None):
	l = list(a)
	l.append(add)
	temp_array = np.array(l)
	if choose_list:
		temp_array = temp_array[choose_list]
	return temp_array

def multiplyData(y,node,edge,edge_intra,y_object,node_object,edge_object,edge_intra_object,edge_intra_object_human):
	
	y_ = []
	y_anticipation_ = []
	node_ = []
	edge_ = []
	edge_intra_ = []
	
	y_object_ = []
	y_object_anticipation_ = []
	node_object_ = []
	edge_object_ = []
	edge_intra_object_ = []
	edge_intra_object_human_ = []

	N = len(node)

	for l in y:
		y_anticipation_.append(appendToArray(l[1:],11))


	for l, n , e, ei, yo, no, eo, eio, eioh in zip(y,node,edge,edge_intra,y_object,node_object,edge_object,edge_intra_object,edge_intra_object_human):
		samples = sampleSubSequences(n.shape[0],extra_samples,min_length_sequence)
		for yo_, no_, eo_, eio_, eioh_ in zip(yo, no, eo, eio, eioh):
			y_object_.append(yo_)
			y_object_anticipation_.append(appendToArray(yo_[1:],13))
			node_object_.append(no_)
			edge_object_.append(eo_)
			edge_intra_object_.append(eio_)
			edge_intra_object_human_.append(eioh_)
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
				edge_intra_.append(ei[ll,:])
				new_list = [(x+1) for x in ll]		
				y_anticipation_.append(appendToArray(l,11,new_list))

				for yo_, no_, eo_, eio_, eioh_ in zip(yo, no, eo, eio, eioh):
					y_object_.append(yo_[ll])
					node_object_.append(no_[ll,:])
					edge_object_.append(eo_[ll,:])
					edge_intra_object_.append(eio_[ll,:])
					edge_intra_object_human_.append(eioh_[ll,:])
					y_object_anticipation_.append(appendToArray(yo_,13,new_list))
			else:
				y_.append(l[s[0]:s[1]])
				node_.append(n[s[0]:s[1],:])
				edge_.append(e[s[0]:s[1],:])
				edge_intra_.append(ei[s[0]:s[1],:])
				new_list = range(s[0]+1,s[1]+1)
				y_anticipation_.append(appendToArray(l,11,new_list))
				for yo_, no_, eo_, eio_, eioh_ in zip(yo, no, eo, eio, eioh):
					y_object_.append(yo_[s[0]:s[1]])
					node_object_.append(no_[s[0]:s[1],:])
					edge_object_.append(eo_[s[0]:s[1],:])
					edge_intra_object_.append(eio_[s[0]:s[1],:])
					edge_intra_object_human_.append(eioh_[s[0]:s[1],:])
					y_object_anticipation_.append(appendToArray(yo_,13,new_list))
			N += 1
	y = y + y_
	edge = edge + edge_
	node = node + node_
	edge_intra = edge_intra + edge_intra_
	y_anticipation = y_anticipation_

	y_object = y_object_
	y_object_anticipation = y_object_anticipation_
	edge_object = edge_object_
	node_object = node_object_
	edge_intra_object = edge_intra_object_
	edge_intra_object_human = edge_intra_object_human_

	return N,y,node,edge,edge_intra,y_object,node_object,edge_object,edge_intra_object,edge_intra_object_human,y_anticipation,y_object_anticipation

if __name__ == '__main__':

	global min_length_sequence, use_data_augmentation, extra_samples, copy_start_state, params, fold, train_activities, test_activities, prefix
	use_data_augmentation = True
	min_length_sequence = 4
	extra_samples = 0 #100
	copy_start_state = True
	params = {
		'use_data_augmentation':use_data_augmentation,
		'min_length_sequence':min_length_sequence,
		'extra_samples':extra_samples,
		'copy_start_state':copy_start_state,
		}

	folds = ['1','2','3','4']	
	prefix = sixDigitRandomNum()	
	for fold in folds:	
		#s='/scr/ashesh/activity-anticipation/features_full_model'
		s='/scail/scratch/group/cvgl/ashesh/activity-anticipation/features_ground_truth'
		test_file = '/scail/scratch/group/cvgl/ashesh/activity-anticipation/activityids_fold{0}.txt'.format(fold)
		
		lines = open(test_file).readlines()
		test_activities = []
		for line in lines:
			line = line.strip()
			if len(line) > 0:
				test_activities.append(line)	
		print "test ",test_file

		train_activities = []
		
		for j in folds:
			if j == fold:
				continue
			train_file = '/scail/scratch/group/cvgl/ashesh/activity-anticipation/activityids_fold{0}.txt'.format(j)
			print "train ",train_file
			lines = open(train_file).readlines()
			for line in lines:
				line = line.strip()
				if len(line) > 0:
					train_activities.append(line)
		print len(train_activities)
		print len(test_activities)
		N = len(train_activities) + len(test_activities)
		print N
		sortActivities(s)
