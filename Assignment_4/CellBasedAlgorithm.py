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

def getDataset(filepath, dataChoice):
	"""
	Parameters
	----------
	filepath : string
		The filepath relative to the parentPath
	dataChoice : int
		Choose the type of data to return
			1: pause_video, play_video
			2: play_video, seek_video
			3: pause_video, seek_video

	Returns
	-------
	list


	"""
	dataDir = dataPath + filepath
	with open(dataDir, 'rb') as csvfile:
		data_col = csv.reader(x.replace('\0', '') for x in csvfile)
		# data_col = csv.reader(csvfile)
		if(dataChoice == 1):
			for data in data_col:
				yield [data[0], data[2], data[3]]	# [user_id, pause_video, play_video]
		if(dataChoice == 2):
			for data in data_col:
				yield [data[0], data[3], data[4]]	# [user_id, play_video, seek_video]
		if(dataChoice == 3):
			for data in data_col:
				yield [data[0], data[2], data[4]]	# [user_id, pause_video, seek_video]

def Euclidean_Distance(p, o):
	"""
		let p and o be a transposed vector
	"""
	_p = np.transpose(p)
	_o = np.transpose(o)
	_dist = _p - _o
	return np.linalg.norm(_dist)

def sumCells(gridCount, index_1, index_2, layer = 0, Ndim = 2):
	_sum = 0
	_lower_bound = 0
	_upper_bound = 0
	if(layer == 0):
		_upper_bound = 1
	elif(layer == 1):
		_lower_bound = 1
		_upper_bound = 2
	elif(layer == 2):
		_lower_bound = 3
		_upper_bound = 4

	for i in range(index_1 - _lower_bound , index_1+ _upper_bound):
		for j in range(index_2 - _lower_bound, index_2 + _upper_bound):
			_sum = gridCount[i][j]
	return _sum

def labelCellsL1(gridStatus, index_1, index_2, Ndim = 2):
	for i in range(index_1 - 1, index_1+ 2):
		for j in range(index_2 - 1, index_2 + 2):
			if( (i == index_1) & (j == index_2)):
				gridStatus[i][j] = 3 # label red
			else:
				if(gridStatus[i][j] == 0):
					gridStatus[i][j] = 2 # label pink

def getL2Cells(gridPoints, index_1, index_2 , Dim1, Dim2):
	_lowerIndex1 = index_1 - 3
	_upperIndex1 = index_1 + 4
	inner_lowerIndex1 = index_1 - 1
	inner_upperIndex1 = index_1 + 1
	_lowerIndex2 = index_2 - 3
	_upperIndex2 = index_2 + 4
	inner_lowerIndex2 = index_2 - 1
	inner_upperIndex2 = index_2 + 1
	_ret = []
	for x in range(_lowerIndex1,_upperIndex1):
			for y in range(_lowerIndex2, _upperIndex2):
				if( (x >= 0) & ( x < Dim1 ) & (y >= 0) & ( y < Dim2 ) 
					& ((x < inner_lowerIndex1) | (x > inner_upperIndex1)) 
					& ((y < inner_lowerIndex2) | (y > inner_upperIndex2))):
						for point in gridPoints["(" + str(x) + "," + str(y) + ")"]:
							_ret.append(point)
	return _ret

def FindAllOutsM(dataset, M, D, dataChoice = 1, Ndim = 2):
	# 3 is red
	# 2 is pink
	# 1 is outlier
	# 0 is white/uncolored
	l = D / np.sqrt(Ndim) / 2
	# for 2 dimensional data
	Min1 = np.min(dataset[:,1])
	Max1 = np.max(dataset[:,1])
	Min2 = np.min(dataset[:,2])
	Max2 = np.max(dataset[:,2])
	Dim1 = int(np.ceil((Max1 - Min1)/ l))
	Dim2 = int(np.ceil((Max2 - Min2)/ l))
	print "Dimension 1: " + str(Dim1)
	print "Dimension 2: " + str(Dim2)
	# TODO: for multidimentional data
	##
	## add padding to the grid to simplify the computation by avoiding edge/boundary cases
	gridCount = np.zeros((Dim1 + 6, Dim2 + 6))
	gridStatus = np.zeros((Dim1 + 6, Dim2 + 6))
	gridPoints = defaultdict(lambda:[])
	pointStatus = np.zeros((len(dataset), 1), dtype = int)
	_outlier = []
	for data in dataset:
		xCoor = np.int(np.float(data[1] - Min1)/ l) + 3 # 3 is the offset gain from padding
		yCoor = np.int(np.float(data[2] - Min2)/ l) + 3 # 3 is the offset gain from padding
		gridCount[xCoor][yCoor] += 1
		gridPoints["(" + str(xCoor) + "," + str(yCoor) + ")"].append(int(data[0]))
	# print gridPoints

	# first scan for unit cell
	for x in xrange(3, Dim1 + 3): # remember the offset
		# print len(grid[:,0])
		for y in xrange(3, Dim2 + 3): # remember the offset
			# print len(grid[0,:])
			countW1 = sumCells(gridCount,x,y,0)
			if(countW1 > M):
				labelCellsL1(gridStatus, x,y)

	# second scan to fill cells depending on L1 cells
	for x in xrange(3, Dim1 + 3): # remember the offset
		# print len(grid[:,0])
		for y in xrange(3, Dim2 + 3): # remember the offset
			# print len(grid[0,:])
			if(gridStatus[x][y] == 0):
				countW2 = sumCells(gridCount,x,y,1)
				if(countW2 > M):
					gridStatus[x][y] = 2 # label as pink
				else:
					countW3 = sumCells(gridCount,x,y,2)
					if(countW3 <= M):
						gridStatus[x][y] = 1 # label as outliers
					else:
						for p in gridPoints["(" + str(x) + "," + str(y) + ")"]:
							countP = countW2
							for q in getL2Cells(gridPoints, index_1, index_2, Dim1, Dim2):
								if(Euclidean_Distance(dataset[p,1:],dataset[q,1:]) <= D):
									countP += 1
									if(countP > M):
										break
							pointStatus[p] = 1
							_outlier.append(p) 

	# labels all the points
	for x in xrange(3, Dim1 + 3): # remember the offset
		# print len(grid[:,0])
		for y in xrange(3, Dim2 + 3): # remember the offset
			_status = gridStatus[x][y]
			if(_status == 0):
				continue
			for point in gridPoints["(" + str(x) + "," + str(y) + ")"]:
				pointStatus[point] = _status
				if(_status == 1):
					_outlier.append(point)

	# for x in xrange(3, Dim1 + 3):
	# 	print gridCount[x]

	return (pointStatus, _outlier)

def countN(pointStatus, n):
	_count = 0
	for i in xrange(len(pointStatus)):
		if(pointStatus[i] == n):
			_count += 1
	return _count

if __name__ == '__main__':
	_dataset = getDataset("click-stream event.csv", 1)
	print _dataset.next()
	dataset = np.array(list(_dataset), dtype = float)
	# print dataset
	# print dataset[:,2]

	# # Understanding the distribution of the dataset using visualisation
	# plt.figure(1)
	# plt.plot(dataset[:,1], dataset[:,2], 'ro')
	# plt.xlabel("pause_video")
	# plt.ylabel("play_video")
	# plt.show()
	pointStatus, outlierList = FindAllOutsM(dataset, 20, 10, 1)
	# print "pointStatus"
	# print pointStatus
	print "number of 0 (special)"
	print countN(pointStatus, 0)
	print "number of 1 (outlier)"
	print countN(pointStatus, 1)
	print "number of 2 (pink)"
	print countN(pointStatus, 2)
	print "number of 3 (red)"
	print countN(pointStatus, 3)

	print "outlierList"
	print outlierList