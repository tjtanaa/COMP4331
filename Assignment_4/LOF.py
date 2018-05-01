from collections import defaultdict
import csv 
import os
from os.path import join
from matplotlib import pyplot as plt
import csv
import numpy as np

parentPath = os.getcwd() + os.path.sep
dataPath = parentPath + "Dataset/"
statsPath = parentPath + "Statistics/"
figPath = parentPath + "Figures/"

def getDataset(filepath):
	"""
	Parameters
	----------
	filepath : string
		The filepath relative to the parentPath

	Returns
	-------
	list


	"""
	dataDir = dataPath + filepath
	with open(dataDir, 'rb') as csvfile:
		data_col = csv.reader(x.replace('\0', '') for x in csvfile)
		# data_col = csv.reader(csvfile)
		if(filepath.find('click-stream') != -1):
			# return list(data_col)
			for data in data_col:
				yield data	# [user_id, load_video, pause_video, play_video, seek_video, speed_change_video, stop_video]

def Euclidean_Distance(p, o):
	"""
		let p and o be a transposed vector
	"""
	_p = np.transpose(p)
	_o = np.transpose(o)
	_dist = _p - _o
	return np.linalg.norm(_dist)

def Manhattan_Distance(p, o):
	"""
		let p and o be a transposed vector
	"""
	_p = np.transpose(p)
	_o = np.transpose(o)
	_dist = np.absolute(_p - _o)
	# return np.sum(np.absolute(p - o))
	return np.sum(_dist)

def reachability_p_o(ith_temp_dist, _kDist, oIndexes):
	_list = np.zeros((len(oIndexes),1))
	for i in xrange(len(oIndexes)):
		_list[i] = np.maximum(_kDist, ith_temp_dist[oIndexes[i]])
	return _list

def lrd_p(_kNNList, reachabilityList):
	return len(_kNNList) / np.sum(reachabilityList)

def compute_LOF(p, _lrdvalues, pth_kNN_List):
	_ret = 0.0
	for i in xrange(len(pth_kNN_List)):
		_ret += _lrdvalues[pth_kNN_List[i]]

	return _ret / len(pth_kNN_List) / _lrdvalues[p]

def _LOF(dataset, k, distance_function = Euclidean_Distance):
	# number of data points
	number_of_data = len(dataset)
	# an array which stores the data calculated to reduce computations
	_temp_dist = np.ones((number_of_data,number_of_data)) * np.inf
	for _data1 in dataset:
		for _data2 in dataset:
			if(int(_data2[0]) > int(_data1[0])):
				_dist = distance_function(np.array(_data1[1:], dtype= float),np.array(_data2[1:], dtype = float))
				_temp_dist[int(_data1[0])][int(_data2[0])] = _dist
				_temp_dist[int(_data2[0])][int(_data1[0])] = _dist

	# to store the LOFs
	_LOFvalues = np.ones((number_of_data,1)) * np.inf

	# to store the lrds
	_lrdvalues = np.ones((number_of_data,1)) *np.inf

	# to store a list of the k-nearest neighbors of p
	_kNN_p = []
	# find k-nearest neighbors of p and its k-distance
	for p in xrange(number_of_data):
		# k-nearest neighbors list
		_kNNList = []
		# k-distance
		_kDist = 0
		# obtain the index of the sorted array
		_index = np.argsort(_temp_dist[p])
		# Start searching or the k-nearest neighbors
		for j in xrange(k):
			_kNNList.append(_index[j])
		# assign k-distance
		_kDist = _temp_dist[p][_index[k - 1]]

		# Check for any other points which are a the same distance as neighbor K
		for j in xrange(k, number_of_data):
			if(np.isclose(_temp_dist[p][j], _kDist)):
				_kNNList.append(_index[j])
			else:
				break

		# compute reachability
		reachabilityList = reachability_p_o(_temp_dist[p], _kDist, _kNNList)

		# compute the local reachability density
		_lrdvalues[p] = lrd_p(_kNNList, reachabilityList)

		# append to the _kNN_p
		_kNN_p.append(_kNNList)

	# compute the LOF values
	for p in xrange(number_of_data):
		_LOFvalues[p] = compute_LOF(p, _lrdvalues, _kNN_p[p])

	return _LOFvalues

def LOF(dataset, k, distance_function = Euclidean_Distance):
	_LOF_list = _LOF(dataset, k, distance_function)
	_outlier = sorted(range(len(_LOF_list)), key=lambda j: _LOF_list[j])[-5:]
	# print _outlier
	for index in _outlier:
		print _LOF_list[index]
	return {"LOF": _LOF_list, "Outlier": _outlier}

if __name__ == '__main__':
	_dataset = getDataset("click-stream event.csv")
	print _dataset.next()
	# print _dataset
	dataset = list(_dataset)
	Manhattan_LOF = LOF(dataset, 2, Manhattan_Distance)
	print "Top 5 Outliers of LOF with distance function (Manhattan_Distance, k = 2) are "
	print Manhattan_LOF["Outlier"]
	Manhattan_LOF = LOF(dataset, 3, Manhattan_Distance)
	print "Top 5 Outliers of LOF with distance function (Manhattan_Distance, k = 3) are "
	print Manhattan_LOF["Outlier"]
	Euclidean_LOF = LOF(dataset, 2, Euclidean_Distance)
	print "Top 5 Outliers of LOF with distance function (Euclidean_Distance, k = 2) are "
	print Euclidean_LOF["Outlier"]
	Euclidean_LOF_2 = LOF(dataset, 3, Euclidean_Distance)
	print "Top 5 Outliers of LOF with distance function (Euclidean_Distance, k = 3) are "
	print Euclidean_LOF_2["Outlier"]

	plt.figure(1)
	index = range(len(dataset))
	plt.plot(index, Manhattan_LOF["LOF"])
	plt.xlabel("Index of Dataset")
	plt.ylabel("LOF values")
	plt.title("Manhattan_LOF")

	plt.figure(2)
	index = range(len(dataset))
	plt.plot(index, Euclidean_LOF["LOF"])
	plt.xlabel("Index of Dataset")
	plt.ylabel("LOF values")
	plt.title("Euclidean_LOF")

	plt.figure(3)
	index = range(len(dataset))
	plt.plot(index, Euclidean_LOF_2["LOF"])
	plt.xlabel("Index of Dataset")
	plt.ylabel("LOF values")
	plt.title("Euclidean_LOF_2")
	plt.show()




