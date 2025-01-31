import scipy.io as sio
import sklearn as sk
from sklearn import ensemble
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.metrics import classification_report
import timeit
import numpy as np
import os
from os.path import join

n_train_samples = 10000
n_test_samples = 1000
parentPath =  os.getcwd() + os.path.sep
outputPath =  parentPath + "Random_Forest/"

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

"""
	This function reports the scores: precision, accuracy, f1 score, recall, training time
	Types of criterion:
	'gini','entropy'

	@param: 
	y_true 	: ground truth
	y_pred	: prediction
	training_time	: time taken to train the model
	criterion : the type of mathematical function to evaluate the information gain
	max_depth : the depth of the tree
	version	: 
"""

def reportScores(y_true, y_pred ,training_time, criterion, max_depth,version=''):
	_accuracy = accuracy_score(y_true, y_pred)
	_f1 = f1_score(y_true,y_pred, average ='weighted') # calculate the f1 scores using the weighted f1 score formula
	_recall = recall_score(y_true,y_pred, average ='weighted') # calculate the recall using the weighted recall formula
	_precision = precision_score(y_true, y_pred, average ='weighted') # calculate the precision using the weighted precision formula

	## the following codes is used to evaluate the precision, recall, accuracy, support of each classes
	# target_names = ['class 0', 'class 1', 'class 2', \
	# 'class 3', 'class 4', 'class 5', \
	# 'class 6', 'class 7', 'class 8', \
	# 'class 9']
	#classification_scores = classification_report(test_labels, test_predicted, target_names=target_names)

	# print classification_scores
	score_metric_file = outputPath + 'Random_Forest_' + str(criterion) + str(max_depth) + '_classification_report' + str(version) +'.txt'
	f = open(score_metric_file, 'w')
	# f.write(str(target_names) + '\t')
	# f.write('\n')
	f.write('Precision \t')
	f.write(str(_precision))
	f.write('\n')
	f.write('Accuracy \t')
	f.write(str(_accuracy))
	f.write('\n')
	f.write('F1 score \t')
	f.write(str(_f1))
	f.write('\n')
	f.write('Recall \t \t')
	f.write(str(_recall))
	f.write('\n')
	f.write('Training Time \t')
	f.write(str(training_time))
	f.write('\n')
	# f.write(classification_scores)
	f.close()
	print "classifier: Random_Forest"
	print "Precision"
	print _precision
	print "Accuracy"
	print _accuracy
	print "f1_score"
	print _f1
	print "Recall"
	print _recall
	print "Training Time"
	print training_time
	print

"""
	Types of criterion:
	'gini','entropy'

	@params
	train_images	: training set samples
	train_labels	: training set labels / ground truth
	test_images		: test set samples
	test_labels		: test set labels
	criterion : the type of mathematical function to evaluate the information gain
	max_depth : the depth of the tree
	version			:
"""

def RForestClassification(train_images, train_labels, test_images, test_labels, criterion = 'gini', max_depth = None,version=''):
	clf = ensemble.RandomForestClassifier(n_estimators=10, criterion=criterion, max_depth=max_depth, min_samples_split=2, min_samples_leaf=1,\
	 min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, \
	 bootstrap=True, oob_score=False, n_jobs=1, random_state=None, verbose=0, warm_start=False, class_weight=None)

	print "start_training"
	start = timeit.default_timer()
	clf = clf.fit(train_images, train_labels)
	"""
		save model as picklefile
	"""
	picklefile = outputPath + 'Random_Forest_' + str(criterion) + str(max_depth) + '_classifier.pkl'
	joblib.dump(clf, picklefile)
	end = timeit.default_timer()
	print "training_ended"
	test_predicted = clf.predict(test_images)
	reportScores(test_labels, test_predicted, end - start, criterion, max_depth, version)

if __name__ == "__main__":
	train_images = getData("Dataset/train_images.mat")
	train_labels = getData("Dataset/train_labels.mat")
	test_images = getData("Dataset/test_images.mat")
	test_labels = getData("Dataset/test_labels.mat")
	RForestClassification(train_images, train_labels, test_images, test_labels,'gini')
	RForestClassification(train_images, train_labels, test_images, test_labels,'entropy')
	RForestClassification(train_images, train_labels, test_images, test_labels,'gini', 5)
	RForestClassification(train_images, train_labels, test_images, test_labels,'entropy', 5)
	RForestClassification(train_images, train_labels, test_images, test_labels,'gini',10)
	RForestClassification(train_images, train_labels, test_images, test_labels,'entropy',10)
