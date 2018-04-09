from sklearn.cluster import DBSCAN
from sklearn import metrics
import matplotlib.pyplot as plt
import scipy.io as sio
import numpy as np
import os
from collections import defaultdict
import timeit

# Assignment Specific User Defined Functions and Variables
parentPath = os.getcwd() + os.path.sep
dataPath = parentPath + "Dataset/"
figPath = parentPath + "images/"

def getDataset(filepath):
	filepath = dataPath + filepath
	if(filepath.find('DBSCAN-Points') != -1):
		return sio.loadmat(filepath)['Points']

def saveFigureAsPNG(filename, fig):
	figDir = figPath + filename + '.png'
	fig.savefig(figDir)

def sklearnDBSCAN(data, eps = 0.12, min_samples = 3):
	db = DBSCAN(eps=eps, min_samples = min_samples).fit(data)
	core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
	core_samples_mask[db.core_sample_indices_] = True
	labels = db.labels_
	print "db.labels_"
	print db.labels_

	# Number of clusters in labes, ignoring noise if present.
	n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

	print "Estimated number of clusters: %d" % n_clusters_

	# Black removed and is used for noise instead.
	unique_labels = set(labels)
	colors = [plt.cm.Spectral(each)
	          for each in np.linspace(0, 1, len(unique_labels))]
	sampleFig = plt.figure(0)
	for k, col in zip(unique_labels, colors):
	    if k == -1:
	        # Black used for noise.
	        col = [0, 0, 0, 1]

	    class_member_mask = (labels == k)

	    xy = data[class_member_mask & core_samples_mask]
	    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
	             markeredgecolor='k', markersize=6)

	    xy = data[class_member_mask & ~core_samples_mask]
	    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
	             markeredgecolor='k', markersize=6)

	plt.title('Estimated number of clusters: %d' % n_clusters_)
	plt.show()
	saveFigureAsPNG("Sklearn_eps" + str(eps) + "_minPts" + str(min_samples), sampleFig)
	return db

####################################################################
def euclideanDistance(p1,p2):
	#Implement the mathematical equation of euclideanDistance

def _inVicinity(ref,p, eps):
	if(euclideanDistance(ref,p) < eps):
		return True
	else:
		return False

def regionQuery(p, eps, data):
	############ These codes can be summarised into one line ##########
	# ret_list = []
	# for index, pt in enumerate(data):
	# 	if _inVicinity(p,pt,eps):
	# 		ret_list.append(index)
	# return ret_list
	##################################################################
	return [index for index, pt in enumerate(data) if _inVicinity(p,pt,eps)]

def expandCluster(p_Index, neighborPtsIndex, C, eps, minPts, ptStatus, ptLabels, data):
	# add p to cluster C
	?
	for ptIndex in neighborPtsIndex:
		# if p' in not visited
		if(ptStatus[ptIndex] == 0):
			# mark p' as visited # visited := 1
			?
			# get the points within [p']'s eps-neighborhood(including p')
			pt_neighborPtsIndex = regionQuery(data[ptIndex], eps, data)

			# if the p' is the core point
			if(len(pt_neighborPtsIndex) >= minPts):
				# prune the repeated core points (core points that exist in the neighbotPtsIndex)
				tempNeighbor = ?
				for pt in tempNeighbor:
					# add to neighborPtsIndex queue
					neighborPtsIndex.append(pt)
					# add to cluster
					ptLabels[pt] = C

		# if p' is the border point
		if(ptLabels[ptIndex] == 0):
			ptLabels[ptIndex] = C

		# if the seed point initially labeled as NOISE
		# it is now added to the cluster
		if(ptLabels[ptIndex] == -1):
			ptLabels[ptIndex] = C

def _user_DBSCAN(data, eps = 0.12, minPts = 3):
	# data structure to keep track of visited nodes
	ptStatus = np.zeros(len(data), dtype = 'int8')
	ptLabels = np.zeros(len(data), dtype = 'int8')

	# initialise the cluster ID to 0
	C = 0
	for pindex, p in enumerate(data):
		? # check whether the point is unvisited
			? # marked as visited
			# return all points within p's eps-neighborhood(including p)
			# return indexes of points within p's eps-neighborhood(including p)
			?
			?# if size of NeighborPoints is less than MinPts
				? # mark pindex as noise # NOISE := -1 
			else:
				? # next cluster
				# index in this function represents the point p
				expandCluster(pindex, neighborPtsIndex, C, eps, minPts, ptStatus, ptLabels, data)
		else: # visited points
			continue
	return ptLabels

def userDBSCAN(data, eps = 0.12, minPts = 3, plotDB = False):
	userDB_Labels = _user_DBSCAN(dbDataset, eps = eps, minPts = minPts)
	# Plot the output of the clusters
	# Number of clusters in labes, ignoring noise if present.
	n_clusters_ = len(set(userDB_Labels)) - (1 if -1 in userDB_Labels else 0)

	print "Estimated number of clusters: %d" % n_clusters_
	# print "Silhouette Coefficient: %f" % metrics.silhouette_score(userDB_Labels,labels)

	# Black removed and is used for noise instead.
	unique_labels = set(userDB_Labels)
	colors = [plt.cm.Spectral(each)
	          for each in np.linspace(0, 1, len(unique_labels))]
	userFig = plt.figure(1)
	for k, col in zip(unique_labels, colors):
	    if k == -1:
	        # Black used for noise.
	        col = [0, 0, 0, 1]

	    # obtain a list of points belong to class k
	    classPoints = [dbDataset[ClassPoint_Index] for ClassPoint_Index, labelPt in enumerate(userDB_Labels) if (labelPt == k)]
	    # convert to numpy array which supports array[:,0] notation
	    arrayClassPoints = np.array(classPoints)
	    # plot points of class k
	    plt.plot(arrayClassPoints[:,0], arrayClassPoints[:,1], 'o', markerfacecolor=tuple(col),
	             markeredgecolor='k', markersize=6)

	plt.title('Estimated number of clusters: %d' % n_clusters_)
	plt.show()
	# save figure
	saveFigureAsPNG("userDBSCAN", userFig)

#########################################################################################

if __name__ == "__main__":
	dbDataset = getDataset("DBSCAN-Points.mat")
	userDBSCAN(dbDataset)

	#### All the code belows are used for debugging  ###############
	#### They are left as it is for future reference #################
	# # testing with sklearn DBSCAN
	# db = sklearnDBSCAN(dbDataset)

	# # Unit Testing euclideanDistance()
	# p1 = [2,3]
	# p2 = [3,9]
	# print euclideanDistance(p1,p2)

	# # This is setup to compare the output of sklearn DBSCAN and userDBSCAN
	# print "sklearnDBSCAN"
	# # convert the labels into user type labels.
	# # Note: in our function label 0 means unvisited. In other words, 0 is not a cluster.
	# for index, _ in enumerate(db.labels_):
	# 	if db.labels_[index] == -1:
	# 		continue
	# 	else:
	# 		db.labels_[index] += 1
	# print db.labels_

	# # # Testing _user_DBSCAN
	# start = timeit.default_timer()
	# print str(start)
	# userDB_Labels = _user_DBSCAN(dbDataset, eps = 0.12, minPts = 3)
	# end = timeit.default_timer()
	# print "Clustering Time"
	# print(str(end - start))
	# print "userDB_Labels"
	# print userDB_Labels
	# print "Verify consistency of label output (length of labels)"
	# print len(userDB_Labels)

	# # Plot the output of the clusters
	# # Number of clusters in labes, ignoring noise if present.
	# n_clusters_ = len(set(userDB_Labels)) - (1 if -1 in userDB_Labels else 0)

	# print "Estimated number of clusters: %d" % n_clusters_
	# # print "Silhouette Coefficient: %f" % metrics.silhouette_score(userDB_Labels,labels)

	# # Black removed and is used for noise instead.
	# unique_labels = set(userDB_Labels)
	# colors = [plt.cm.Spectral(each)
	#           for each in np.linspace(0, 1, len(unique_labels))]
	# userFig = plt.figure(1)
	# for k, col in zip(unique_labels, colors):
	#     if k == -1:
	#         # Black used for noise.
	#         col = [0, 0, 0, 1]

	#     # obtain a list of points belong to class k
	#     classPoints = [dbDataset[ClassPoint_Index] for ClassPoint_Index, labelPt in enumerate(userDB_Labels) if (labelPt == k)]
	#     # convert to numpy array which supports array[:,0] notation
	#     arrayClassPoints = np.array(classPoints)
	#     # plot points of class k
	#     plt.plot(arrayClassPoints[:,0], arrayClassPoints[:,1], 'o', markerfacecolor=tuple(col),
	#              markeredgecolor='k', markersize=6)

	# plt.title('Estimated number of clusters: %d' % n_clusters_)
	# plt.show()
	# # save figure
	# saveFigureAsPNG("userDBSCAN", userFig)


