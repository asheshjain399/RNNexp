import numpy as np
import sys
import cPickle
import threading
from evaluateCheckpoint import evaluate, evaluateForAllThresholds

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
		else:
			continue


	return checkpoints,threshold,dataset_id

maneuver_type = sys.argv[1]
model_id = sys.argv[2]

model_file = 'checkpoints/{0}/{1}/model'.format(maneuver_type,model_id)

checkpoints,threshold,dataset_id = readModelFile(model_file)

print "Checkpoints ",checkpoints
print "Prediction threshold ",threshold
print "dataset index ",dataset_id
with open('settings.py','w') as f:
	f.write('OUTPUT_THRESH = %f \n' % threshold)

precision = []
recall = []
anticipation_time = []
	
folds = ['fold_1', 'fold_2', 'fold_3', 'fold_4', 'fold_5']
for fold,checkpoint in zip(folds,checkpoints):
	path_to_dataset = 'checkpoints/{0}/{1}/test_data_{2}.pik'.format(maneuver_type,fold,dataset_id)
	path_to_checkpoint = 'checkpoints/{0}/{1}/{2}/checkpoint.{3}'.format(maneuver_type,model_id,fold,checkpoint)
	conMat,p_mat,re_mat,time_mat = evaluate(path_to_dataset,path_to_checkpoint)
	precision.append(np.mean(np.diag(p_mat)[1:]))
	recall.append(np.mean(np.diag(re_mat)[1:]))
	anticipation_time.append(np.mean(  np.divide(np.diag(time_mat)[1:],np.diag(conMat)[1:])   ))

precision = np.array(precision)
recall = np.array(recall)
anticipation_time = np.array(anticipation_time)

print "Pr={0:.4f}({1:.4f})  Re={2:.4f}({3:.4f}) t={4:.4f}({5:.4f})".format(np.mean(precision),np.std(precision),np.mean(recall),np.std(recall),np.mean(anticipation_time),np.std(anticipation_time))
