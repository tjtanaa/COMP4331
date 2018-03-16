from sklearn.externals import joblib
import scipy.io as sio
import numpy as np

# Assignment Specific User Defined Functions and Variables
parentPath = "D:\SPRING 2018\COMP 4331 sit\Assignment\Assignment_2\\"
DtreePath = parentPath + "Decision_Tree\\"
KNNPath = parentPath + "KNN\\"
SVMPath = parentPath + "SVM\\"
RForestPath = parentPath + "Random_Forest\\"
MLPPath = parentPath + "MLP\\"
DSplitPath = parentPath + "DataSplit\\"

n_train_samples = 10000
n_test_samples = 1000

"""
	this is the function that is going to read the mat file and return a numpy array
	this function will check for the keywords to determine which dataset to be imported
"""

# this is the function that is going to read the mat file and return a numpy array
def getData(filepath):
	if(filepath.find('train_images') != -1):
		filepath = parentPath + filepath
		return sio.loadmat(filepath)['train_images']

	if(filepath.find('train_labels') !=-1):
		filepath = parentPath + filepath
		return np.reshape(sio.loadmat(filepath,)['train_labels'], (n_train_samples,1))

	if(filepath.find('test_images') !=-1):
		filepath = parentPath + filepath
		return sio.loadmat(filepath)['test_images'] 

	if(filepath.find('test_labels') !=-1):
		filepath = parentPath + filepath
		return np.reshape(sio.loadmat(filepath)['test_labels'], (n_test_samples,1))

def load_model(pickleFile):
	# load the model from disk
	return joblib.load(pickleFile)
