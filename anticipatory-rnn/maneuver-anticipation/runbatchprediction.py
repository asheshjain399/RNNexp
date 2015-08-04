import numpy as np
import sys
import cPickle
import threading
from evaluateCheckpoint import evaluate

'''
def worker_thread(path_to_checkpoint,path_to_dataset,checkpoint,checkpoints_params):
	global results_mat_precision, results_mat_recall, results_mat_time 

	count_checkpoint = np.where(checkpoints_params == checkpoint)
	count_checkpoint = count_checkpoint[0][0] 

	[conMat,p_mat,re_mat,time_mat] = evaluate(path_to_dataset,path_to_checkpoint)
	precision = np.mean(np.diag(p_mat)[1:])
	recall = np.mean(np.diag(re_mat)[1:])
	anticipation_time = np.mean(  np.divide(np.diag(time_mat)[1:],np.diag(conMat)[1:])   )
	results_mat_precision[count_th,count_checkpoint,count_fold] = precision
	results_mat_recall[count_th,count_checkpoint,count_fold] = recall
	results_mat_time[count_th,count_checkpoint,count_fold] = anticipation_time
'''


maneuver_type = sys.argv[1]

index = '0'

if maneuver_type == 'all':
	index = '356988'
elif maneuver_type == 'lane':
	index = '723759'
elif maneuver_type == 'turns':
	index = '209221'
else:
	print 'Maneuver mis-match'

folds = ['fold_1', 'fold_2', 'fold_3', 'fold_4', 'fold_5']

checkpoints_params = np.append(np.arange(200, 599, 50), 599)

thresh_params = np.arange(.6, .9, .01)



#global count_th, count_fold, results_mat_precision, results_mat_recall, results_mat_time 

checkpoint_dir = '/scr/ashesh/brain4cars/checkpoints'

results_mat_precision = np.zeros((thresh_params.shape[0],checkpoints_params.shape[0],len(folds)+1))
results_mat_recall = np.zeros((thresh_params.shape[0],checkpoints_params.shape[0],len(folds)+1))
results_mat_time = np.zeros((thresh_params.shape[0],checkpoints_params.shape[0],len(folds)+1))

count_th = 0
for th in thresh_params:
	
	print "Generating results for threshold={0}".format(th)
	with open('settings.py','w') as f:
		f.write('OUTPUT_THRESH = %f \n' % th)

	count_fold = 0
	for fold in folds:
		count_checkpoint = 0
		threads=[]
		for checkpoint in checkpoints_params:
			
			path_to_dataset = 'checkpoints/{0}/{1}/test_data_{2}.pik'.format(maneuver_type,fold,index)
			path_to_checkpoint = '{0}/{1}/{2}/checkpoint.{3}'.format(checkpoint_dir,fold,index,checkpoint)
			
			#t = threading.Thread(target=worker_thread, args=(path_to_checkpoint,path_to_dataset,checkpoint,checkpoints_params,)) 
			#threads.append(t)
			
			[conMat,p_mat,re_mat,time_mat] = evaluate(path_to_dataset,path_to_checkpoint)
			precision = np.mean(np.diag(p_mat)[1:])
			recall = np.mean(np.diag(re_mat)[1:])
			anticipation_time = np.mean(  np.divide(np.diag(time_mat)[1:],np.diag(conMat)[1:])   )
			results_mat_precision[count_th,count_checkpoint,count_fold] = precision
			results_mat_recall[count_th,count_checkpoint,count_fold] = recall
			results_mat_time[count_th,count_checkpoint,count_fold] = anticipation_time
		
			count_checkpoint += 1
			
			'''
			threads[-1].start()
			print 'starting thread'
			'''
		'''	
		for t in threads:
			t.join()
			print 'joining'
		'''

		count_fold += 1
	
	results_mat_recall[count_th,:,-1] = np.mean(results_mat_recall[count_th,:,:-1],axis=1)
	results_mat_precision[count_th,:,-1] = np.mean(results_mat_precision[count_th,:,:-1],axis=1)
	results_mat_time[count_th,:,-1] = np.mean(results_mat_time[count_th,:,:-1],axis=1)

	count_th += 1
	
results = {}
results['precision'] = results_mat_precision
results['recall'] = results_mat_recall
results['time'] = results_mat_time
results['threshold'] = thresh_params
results['checkpoints'] = checkpoints_params

cPickle.dump(results, open('checkpoints/{0}/complete_results_final_model.pik'.format(maneuver_type), 'wb'))
