from utility import load_SVM, getData
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

if __name__ == "__main__":
	# train_images = getData("Dataset\\train_images.mat")
	# train_labels = getData("Dataset\\train_labels.mat")
	test_images = getData("Dataset/test_images.mat")
	test_labels = getData("Dataset/test_labels.mat")
	model = load_SVM()
	y_pred = model.predict(test_images)

	print "Precision"
	print precision_score(test_labels, y_pred,average ='micro')
	print "Recall"
	print recall_score(test_labels, y_pred, average ='micro')
