import numpy as np

def convertToSingleVec(X,new_idx,featureRange):
	keys = X.keys()
	[T,N,D]  = X[keys[0]].shape
	D = len(new_idx) - len(np.where(new_idx < 0)[0])
	single_vec = np.zeros((T,N,D),dtype=np.float32)
	for k in keys:
		nm = k.split(':')[0]
		idx = new_idx[featureRange[nm]]
		insert_at = np.delete(idx,np.where(idx < 0))
		single_vec[:,:,insert_at] = X[k]
	return single_vec

