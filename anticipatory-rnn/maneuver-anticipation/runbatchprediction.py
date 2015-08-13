import numpy as np
import sys
import cPickle
import threading
from evaluateCheckpoint import evaluate, evaluateForAllThresholds

'''
Input:
maneuver_type: {all,lane,turns,all_new_features}

Before running this script make sure that the index of dataset, checkpoint_dir, and path_to_dataset are correctly defined.

Output:
This is an evaluation script. It will evaluate the model for a number of checkpoints, 
and threshold values on all 5 folds. The results will be dumped in a pickle file. 
The user can run analyzeResults.py after this to choose the best threshold and checkpoint values. 
'''
maneuver_type = sys.argv[1]

architectures = ['lstm_one_layer','lstm_two_layers','multipleRNNs']
model_type = 2

index = '0'

if maneuver_type == 'all':
	index = '356988'
elif maneuver_type == 'lane':
	index = '723759'
elif maneuver_type == 'turns':
	index = '209221'
elif maneuver_type == 'all_new_features':
# New features obtained from Avi. AAM features and driver head pose features
	index = '846483'
else:
	print 'Maneuver mis-match'

folds = ['fold_1', 'fold_2', 'fold_3', 'fold_4', 'fold_5']

checkpoints_params = np.append(np.arange(200, 599, 50), 599)

thresh_params = np.arange(.6, .9, .01)

checkpoint_dir = '/scr/ashesh/brain4cars/checkpoints'
#checkpoint_dir = 'checkpoints'

results_mat_precision = np.zeros((thresh_params.shape[0],checkpoints_params.shape[0],len(folds)+1))
results_mat_recall = np.zeros((thresh_params.shape[0],checkpoints_params.shape[0],len(folds)+1))
results_mat_time = np.zeros((thresh_params.shape[0],checkpoints_params.shape[0],len(folds)+1))

count_th = 0
count_fold = 0
for fold in folds:
	count_checkpoint = 0
	threads=[]
	for checkpoint in checkpoints_params:
		path_to_dataset = 'checkpoints/{0}/{1}/test_data_{2}.pik'.format(maneuver_type,fold,index)
		path_to_checkpoint = '{0}/{1}/{2}/checkpoint.{3}'.format(checkpoint_dir,fold,index,checkpoint)
		
		precision,recall,anticipation_time = evaluateForAllThresholds(path_to_dataset,path_to_checkpoint,thresh_params,architectures[model_type])
		results_mat_precision[:,count_checkpoint,count_fold] = precision
		results_mat_recall[:,count_checkpoint,count_fold] = recall
		results_mat_time[:,count_checkpoint,count_fold] = anticipation_time
		count_checkpoint += 1
	
	count_fold += 1

for count_th in range(len(thresh_params)):
	results_mat_recall[count_th,:,-1] = np.mean(results_mat_recall[count_th,:,:-1],axis=1)
	results_mat_precision[count_th,:,-1] = np.mean(results_mat_precision[count_th,:,:-1],axis=1)
	results_mat_time[count_th,:,-1] = np.mean(results_mat_time[count_th,:,:-1],axis=1)
	
results = {}
results['precision'] = results_mat_precision
results['recall'] = results_mat_recall
results['time'] = results_mat_time
results['threshold'] = thresh_params
results['checkpoints'] = checkpoints_params

cPickle.dump(results, open('checkpoints/{0}/complete_results_final_model.pik'.format(maneuver_type), 'wb'))
