import sys
import numpy as np
import theano
import cPickle
from neuralmodels.utils import writeMatToCSV
from neuralmodels.utils import readCSVasFloat
import os
import socket as soc

def unNormalizeData(normalizedData,data_mean,data_std,dimensions_to_ignore):
	T = normalizedData.shape[0]
	D = data_mean.shape[0]
	origData = np.zeros((T,D),dtype=np.float32)
	dimensions_to_use = []
	for i in range(D):
		if i in dimensions_to_ignore:
			continue
		dimensions_to_use.append(i)
	dimensions_to_use = np.array(dimensions_to_use)

	if not len(dimensions_to_use) == normalizedData.shape[1]:
		return []
		
	origData[:,dimensions_to_use] = normalizedData
	
	stdMat = data_std.reshape((1,D))
	stdMat = np.repeat(stdMat,T,axis=0)
	meanMat = data_mean.reshape((1,D))
	meanMat = np.repeat(meanMat,T,axis=0)
	origData = np.multiply(origData,stdMat) + meanMat
	return origData

