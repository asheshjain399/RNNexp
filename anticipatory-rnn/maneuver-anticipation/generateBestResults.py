import numpy as np
import sys
import cPickle
import threading
from evaluateCheckpoint import evaluate, evaluateForAllThresholds
from utils import writeconfmatTofile
'''
Input: 
maneuver_type : {all,lane,turns}
model_id : {model_1, model_2,...}
python generateBestResults.py all model_1
'''

def readModelFile(model_file):
	f = open(model_file)
	lines = f.readlines()
	
	dataset_id = ''
	checkpoints = []
	threshold = 0.0
	model = 'multipleRNNs'

	for line in lines:
		line = line.strip()

		if len(line) == 0:
			continue 
		if line[0] in ['#','%']:
			continue
		if line[:2] == 'id':
			dataset_id = line.split('=')[1]
		elif line[:2] == 'th':
			threshold = float(line.split('=')[1])
		elif line[:4] == 'fold':
			checkpoints.append(line.split(',')[1])
		elif line[:5] == 'model':
			model = line.split('=')[1]
		else:
			continue


	return checkpoints,threshold,dataset_id,model

def getResults(conMat,p_mat,re_mat,time_mat):
	return np.mean(np.diag(p_mat)[1:]),np.mean(np.diag(re_mat)[1:]),np.mean(np.divide(np.diag(time_mat)[1:],np.diag(conMat)[1:])), (np.sum(conMat[1:,0])/np.sum(conMat[:,0]))

def pathToDataset(maneuver_type,fold,dataset_id):
	return 'checkpoints/{0}/{1}/test_data_{2}.pik'.format(maneuver_type,fold,dataset_id)

def pathToCheckpoint(maneuver_type,model_id,fold,checkpoint):
	return 'checkpoints/{0}/{1}/{2}/checkpoint.{3}'.format(maneuver_type,model_id,fold,checkpoint)

def F1Plot(maneuver_type,checkpoints,model_id,model_type,dataset_id):

	thresh_params = np.arange(.15, 1.0, .05)
	folds = ['fold_1', 'fold_2', 'fold_3', 'fold_4', 'fold_5']
	F1 = np.zeros((thresh_params.shape[0],len(folds)))
	fid = 0
	for fold,checkpoint in zip(folds,checkpoints):
		path_to_dataset = pathToDataset(maneuver_type,fold,dataset_id)
		path_to_checkpoint = pathToCheckpoint(maneuver_type,model_id,fold,checkpoint)
		precision,recall,anticipation_time = evaluateForAllThresholds(path_to_dataset,path_to_checkpoint,thresh_params,model_type)
		f1_score = 2.0*precision*recall / (precision+recall)
		F1[:,fid] = f1_score
		fid += 1

	F1 = np.mean(F1,axis=1)
	st = ''
	for x in F1:
		st += str(100.0*x) + ','
	st = st[:-1]

	print "************************"
	print "F1-score"
	print F1
	print st
	print "Threshold values"
	print thresh_params
	print "************************"


def getBestResults(maneuver_type,checkpoints,model_id,model_type,dataset_id):
	folds = ['fold_1', 'fold_2', 'fold_3', 'fold_4', 'fold_5']

	keys = range(10)
	keys.append('best')
	precision = {}
	false_pos_rate = {}
	recall = {}
	anticipation_time = {}
	p_confMat = {}

	for k in keys:
		precision[k] = []
		false_pos_rate[k] = []
		recall[k] = []
		anticipation_time[k] = []
		p_confMat[k] = []	


	for fold,checkpoint in zip(folds,checkpoints):
		path_to_dataset = pathToDataset(maneuver_type,fold,dataset_id)
		path_to_checkpoint = pathToCheckpoint(maneuver_type,model_id,fold,checkpoint)
		conMat,p_mat,re_mat,time_mat = evaluate(path_to_dataset,path_to_checkpoint,model_type)

		for k in conMat.keys():
			if len(conMat[k]) == 0:
				continue
			p_,r_,t_,fp_ = getResults(conMat[k],p_mat[k],re_mat[k],time_mat[k])
			precision[k].append(100.0*p_)
			false_pos_rate[k].append(100.0*fp_)
			recall[k].append(100.0*r_)
			anticipation_time[k].append(t_)
			if len(p_confMat[k]) == 0:
				p_confMat[k] = p_mat[k]
			else:
				p_confMat[k] += p_mat[k]

	for k in keys:

		precision[k] = np.array(precision[k])
		recall[k] = np.array(recall[k])
		anticipation_time[k] = np.array(anticipation_time[k])
		false_pos_rate[k] = np.array(false_pos_rate[k])

	keys = range(10)
	keys = keys[::-1]
	keys.append('best')
	F1_list = []
	pr = 0.0
	re = 0.0
	fpr = 0.0
	ant_time = 0.0
	pr_std = 0.0
	re_std = 0.0
	fpr_std = 0.0
	ant_time_std = 0.0
	for k in keys:
		if len(precision[k]) == 0:
			continue
		pr = np.mean(precision[k])
		re = np.mean(recall[k])
		fpr = np.mean(false_pos_rate[k])
		ant_time = np.mean(anticipation_time[k])

		pr_std_err = np.std(precision[k])/np.sqrt(len(precision[k]))
		fpr_std_err = np.std(false_pos_rate[k])/np.sqrt(len(false_pos_rate[k]))
		re_std_err = np.std(recall[k])/np.sqrt(len(precision[k]))
		ant_time_std = np.std(anticipation_time[k])

		f1 = 2.0*pr*re/(pr+re)
		F1_list.append(f1)
	print precision['best']
	print "****************"
	print "Best results"
	print "Pr={0:.4f}({1:.4f})  Re={2:.4f}({3:.4f}) t={4:.4f}({5:.4f}) fpr={6:.4f}({7:.4f})".format(pr,pr_std_err,re,re_std_err,ant_time,ant_time_std,fpr,fpr_std_err)
	print "****************"
	print "F1-score by varying time to maneuver"
	print F1_list[:-1]
	print "****************"


if __name__ == "__main__":

	maneuver_type = sys.argv[1]
	model_id = sys.argv[2]
	model_file = 'checkpoints/{0}/{1}/model'.format(maneuver_type,model_id)
	checkpoints,threshold,dataset_id,model_type = readModelFile(model_file)
	print "Checkpoints ",checkpoints
	print "Prediction threshold ",threshold
	print "dataset index ",dataset_id
	with open('settings.py','w') as f:
		f.write('OUTPUT_THRESH = %f \n' % threshold)

	exp_type = ''

	if len(sys.argv) == 4:
		exp_type = sys.argv[3]
	
	if exp_type == 'f1':
		F1Plot(maneuver_type,checkpoints,model_id,model_type,dataset_id)
	elif exp_type == 'best':
		getBestResults(maneuver_type,checkpoints,model_id,model_type,dataset_id)
	else:
		F1Plot(maneuver_type,checkpoints,model_id,model_type,dataset_id)
		getBestResults(maneuver_type,checkpoints,model_id,model_type,dataset_id)

	'''
	p = 0.2*p_confMat
	print p
	p = p[[1,2,3,4,0],:]
	p = p[:,[1,2,3,4,0]]
#pt = np.zeros((p.shape[0],p.shape[1]))
#pt[:-1,:] = p[1:,:]
#pt[-1,:] = p[0,:]
	print p
	p = np.transpose(p)
	print p
	writeconfmatTofile(p,'all.csv',['L Lane','R Lane','L Turn','R Turn','Straight'])
	'''
