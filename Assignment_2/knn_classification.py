import scipy.io as sio
import sklearn as sk
from sklearn import neighbors
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, log_loss
from sklearn.metrics import classification_report
import timeit
import numpy as np
import matplotlib.pyplot as plt
import os
from os.path import join


n_train_samples = 10000
n_test_samples = 1000
parentPath =  os.getcwd() + os.path.sep
outputPath =  parentPath + "KNN/"

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

	@param: 
	y_true 	: ground truth
	y_pred	: prediction
	training_time	: time taken to train the model
	n_neighbors	: n number of votes
	version	: 
"""

def reportScores(y_true, y_pred ,training_time, n_neighbors,version=''):
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
	score_metric_file = outputPath + 'KNN_' + str(n_neighbors) + 'classification_report' + str(version) +'.txt'
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
	print "classifier with number of neighbors: " +str(n_neighbors)
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
	Type of distance measurement:
	1 - manhattan_distance(l1)
	2 - Euclidean_distance(l2)
	other - minkowski_distance(p)

	@params
	train_images	: training set samples
	train_labels	: training set labels / ground truth
	test_images		: test set samples
	test_labels		: test set labels
	n_neighbors		: number of sample votes
	p 				: the type of distance measurement
	version			:
"""

def KNNClassification(train_images, train_labels, test_images, test_labels, n_neighbors = 5, p = 2,version=''):
	clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, weights='uniform',algorithm='auto', leaf_size=30, p=p)
	print "start_training"
	start = timeit.default_timer()
	clf = clf.fit(train_images, train_labels) # train the model
	"""
		save model as picklefile
	"""
	picklefile = outputPath + 'KNN_' +  str(n_neighbors) + '.pkl'
	joblib.dump(clf, picklefile) # save the model into a pickle file
	end = timeit.default_timer()
	print "training_ended"
	test_predicted = clf.predict(test_images)
	reportScores(test_labels, test_predicted, end - start, n_neighbors, version)

def findBestK(train_images, train_labels, test_images, test_labels, n_neighbors = 5, p = 2):
	clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, weights='uniform',algorithm='auto', leaf_size=30, p=p)
	print "start_training" + str(n_neighbors)
	# start = timeit.default_timer()
	clf = clf.fit(train_images, train_labels) # train the model
	"""
		save model as picklefile
	"""
	# picklefile = outputPath + 'KNN_' +  str(n_neighbors) + '.pkl'
	# joblib.dump(clf, picklefile) # save the model as a pickle file
	# end = timeit.default_timer()
	print "training_ended"
	test_predicted = clf.predict_proba(test_images) # return prediction and probability
	# print "test_predicted"
	# test_predicted = np.reshape(test_predicted,(n_test_samples,1))
	# print test_predicted
	# print "test_labels"
	# print test_labels
	return log_loss(test_labels, test_predicted) # calculate log losses


if __name__ == "__main__":
	train_images = getData("Dataset/train_images.mat")
	train_labels = getData("Dataset/train_labels.mat")
	test_images = getData("Dataset/test_images.mat")
	test_labels = getData("Dataset/test_labels.mat")

	"""
		This section is the code to plot out the loss function and use elbow method 
	"""
	# print test_images.ravel()
	# lossList = []
	# loss_0 = findBestK(train_images, train_labels, test_images, test_labels, 5)
	# lossList.append(loss_0)
	# index = 0
	# print "Finding best K using Elbow Method"
	# start_finding = timeit.default_timer()
	# while index < 15:
	# 	_loss = findBestK(train_images, train_labels, test_images, test_labels, index*5 + 10)
	# 	lossList.append(_loss)
	# 	index += 1
	# end_finding = timeit.default_timer()
	# print "Done Finding"
	# KNN_Training_Time_file = output + "Training_Time_KNN.txt"
	# f = open(KNN_Training_Time_file, 'w')
	# f.write(str(end_finding - start_finding))
	# f.close()
	# plt.plot(lossList)
	# plt.title('KNN log_loss versus K neighbors ( i*5 + 10)')
	# plt.ylabel('Loss')
	# plt.show()
	"""
		This is after determining the best K and change the K value manually
	"""
	print "Training with Best K"
	KNNClassification(train_images, train_labels, test_images, test_labels, 30)

