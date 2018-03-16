from knn_classification import KNNClassification
from svm_classification import SVMClassification
from utility import getData
import numpy as np
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
	x1 = getData("Dataset\\train_images.mat")
	x2 = getData("Dataset/test_images.mat")
	y1 = getData("Dataset\\train_labels.mat")
	y2 = getData("Dataset/test_labels.mat")
	FullImages = np.concatenate((x1,x2))
	FullLabels = np.concatenate((y1,y2))
	train_images, test_images,train_labels, test_labels = train_test_split(FullImages, FullLabels, test_size=1000, random_state=42)
	KNNClassification(train_images, train_labels, test_images, test_labels, 30, 2, 'new')
	SVMClassification(train_images, train_labels, test_images, test_labels,'linear', 'new')