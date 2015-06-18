import random
import numpy as np

def sixDigitRandomNum():	
	st = ''
	for i in range(6):
		st = st + str(random.randint(0,9))
	return st

def confusionMat(P,Y):
	size = np.max(Y) + 1
	confMat = np.zeros((size,size))
	for p,y in zip(P,Y):
		confMat[p,y] += 1.0
	col_sum = np.reshape(np.sum(confMat,axis=1),(size,1))
	row_sum = np.reshape(np.sum(confMat,axis=0),(1,size))
	precision_confMat = confMat/np.repeat(col_sum,size,axis=1)
	recall_confMat = confMat/np.repeat(row_sum,size,axis=0)
	return confMat,precision_confMat,recall_confMat
