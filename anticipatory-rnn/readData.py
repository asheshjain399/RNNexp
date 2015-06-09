import numpy as np
import re
from os import listdir

def readFeatures(ll):
	colon_seperated = [x.strip() for x in ll.strip().spilt(' ')]
	f_list = [int(x.split(':')[1]) for x in colon_seperated]
	return f_list

def parseColonSeperatedFeatures(colon_seperated):
	f_list = [int(x.split(':')[1]) for x in colon_seperated]
	return f_list

def parseSegment(filename):
	f = open(filename,'r')
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

def parseTemporalEdge(filename):
	f = open(filename,'r')
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


def readActivity(files):	
	input_features_node = []
	input_features_edge = []
	Y = []
	for f in files:
		if len(f.split('_')) == 2:
			key_value = parseSegment(f)
			input_features_node.append(key_value['sub_activity_features'])
			Y.append(key_value['sub_activity'])
		elif len(f.split('_')) == 3:
			key_value = parseTemporalEdge(f)
			input_features_edge.append(key_value['s_s_features'])

		input_features_edge = [ [0]*len(input_features_edge[0]), input_features_edge]

	# only subactivity features
	Y = np.array(Y)
	input_features_node = np.array(input_features_node)
	input_features_edge = np.array(input_features_edge)

	return Y, input_features_node, input_features_edge

def sortActivities(folder):
	Y = []
	features = []

	all_the_files = listdir(folder)

	all_activities = []
	for f in all_the_files:
		s = f.split('_')[0]
		if s not in all_activities:
			all_activities.append(s)

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
	
		y_,node_,edge_ = readActivity(f)

		Y,features = processFeatures(y_,node_,edge_)
