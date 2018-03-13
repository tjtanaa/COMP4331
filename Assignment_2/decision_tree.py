import scipy.io as sio
import sklearn as sk
from sklearn import tree
from sklearn.externals import joblib
from sklearn.metrics import classification_report
import timeit
import numpy as np

n_train_samples = 10000
n_test_samples = 1000
parentPath =  "D:\SPRING 2018\COMP 4331 sit\Assignment\Assignment_2\\"
outputPath =  parentPath + "Decision_Tree\\"

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

def DTreeClassification(train_images, train_labels, test_images, test_labels,criterion = 'gini', max_depth = 5):
	clf = tree.DecisionTreeClassifier(criterion=criterion, \
		 splitter='best', \
		 max_depth= max_depth) #, \
		 # min_samples_split=2, \
		 # min_samples_leaf=1, \
		 # min_weight_fraction_leaf=0.0, \
		 # max_features=None, \
		 # random_state=None, \
		 # max_leaf_nodes=None, \
		 # min_impurity_decrease=0.0, \
		 # min_impurity_split=None, \
		 # class_weight=None, \
		 # presort=False)
	print "start_training"
	start = timeit.default_timer()
	clf = clf.fit(train_images, train_labels)
	picklefile = outputPath + 'decision_tree_' + str(criterion) +'_' + str(max_depth) + '.pkl'
	joblib.dump(clf, picklefile)
	end = timeit.default_timer()
	print "training_ended"

	timefile = outputPath + 'decision_tree_'+ str(criterion) +'_' + str(max_depth) + '_timetake.txt'
	f = open(timefile, 'w')
	f.write(str(end - start))
	f.close()
	test_predicted = clf.predict(test_images)
	target_names = ['class 0', 'class 1', 'class 2', \
	'class 3', 'class 4', 'class 5', \
	'class 6', 'class 7', 'class 8', \
	'class 9']
	classification_scores = classification_report(test_labels, test_predicted, target_names=target_names)

	print classification_scores
	score_metric_file = outputPath + 'decision_tree_' + str(criterion) +'_' + str(max_depth) + 'classification_report.txt'
	f = open(score_metric_file, 'w')
	f.write(classification_scores)
	f.close()

if __name__ == "__main__":
	train_images = getData("Dataset\\train_images.mat")
	train_labels = getData("Dataset\\train_labels.mat")
	test_images = getData("Dataset/test_images.mat")
	test_labels = getData("Dataset/test_labels.mat")
	DTreeClassification(train_images, train_labels, test_images, test_labels, 'gini',5)
	DTreeClassification(train_images, train_labels, test_images, test_labels, 'gini',10)
	DTreeClassification(train_images, train_labels, test_images, test_labels, 'entropy',5)
	DTreeClassification(train_images, train_labels, test_images, test_labels, 'entropy',10)
