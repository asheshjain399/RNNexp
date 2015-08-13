import random
import numpy as np

def sixDigitRandomNum():	
	st = ''
	for i in range(6):
		st = st + str(random.randint(0,9))
	return st

def confusionMat(P,Y,T):
	size = np.max(Y) + 1
	confMat = np.zeros((size,size))
	TimeMat = np.zeros((size,size))
	for p,y,t in zip(P,Y,T):
		confMat[p,y] += 1.0
		TimeMat[p,y] += t
	col_sum = np.reshape(np.sum(confMat,axis=1),(size,1))
	row_sum = np.reshape(np.sum(confMat,axis=0),(1,size))
	precision_confMat = confMat/np.repeat(col_sum,size,axis=1)
	recall_confMat = confMat/np.repeat(row_sum,size,axis=0)
	return confMat,precision_confMat,recall_confMat,TimeMat

def writeconfmatTofile(M,filename,labels):
	f = open(filename,'w')
	st = ''
	for l in labels:
		st += l + ','
	st = st[:-1] + '\n'	
	f.write(st)

	for row in M:
		st = ''
		for ele in row:
			st += str(ele) + ','
		st = st[:-1] + '\n'
		f.write(st)
	f.close()

