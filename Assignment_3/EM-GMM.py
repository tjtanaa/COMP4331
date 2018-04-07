from sklearn.cluster import DBSCAN
from sklearn import metrics
import matplotlib.pyplot as plt
import scipy.io as sio
import numpy as np
import os
from collections import defaultdict
import timeit
from numpy.linalg import inv, det

# Assignment Specific User Defined Functions and Variables
parentPath = os.getcwd() + os.path.sep
dataPath = parentPath + "Dataset/"
figPath = parentPath + "images/"

def getDataset(filepath):
	filepath = dataPath + filepath
	if(filepath.find('GMM-Points') != -1):
		return sio.loadmat(filepath)['Points']

def saveFigureAsPNG(filename, fig):
	figDir = figPath + filename + '.png'
	fig.savefig(figDir)

def MultiGaussianProbability(x, mu, var):
	# Note x is a 2d Point, mu is a 2d array, sigma is also a 2x2 dimentional array
	# Note multivariate
	return np.multiply(1.0/(np.sqrt(2*np.pi*det(var))),np.exp((np.matmul(np.transpose((np.subtract(x,mu))),np.matmul(inv(var),np.subtract(x,mu))))/(-2)))

def score_samples(samples, mu, var):
	ret_array = np.zeros(len(samples))
	for i, sample in enumerate(samples):
		ret_array[i] = MultiGaussianProbability(np.reshape(sample, (2,1)), mu, var)[0][0]
	return ret_array

def UniGaussianProbability(x, mu, var):
	return 1.0/(np.sqrt(2*np.pi*var))*np.exp(-np.square(x - mu)/var)

def plotDataSet(dataset, mu_c, var_c, n_clusters, epoch, title):
	# Plot the update dataset and labels
	unique_labels = set(dataset[:,2])
	colors = [plt.cm.Spectral(each)
	          for each in np.linspace(0, 1, len(unique_labels))]

	userFig = plt.figure(epoch)
	for k, col in zip(unique_labels, colors):

	    # obtain a list of points belong to class k
	    classPoints = [datapoints[0:2] for datapoints in dataset if (datapoints[2] == k)]
	    # convert to numpy array which supports array[:,0] notation
	    arrayClassPoints = np.array(classPoints)
	    # plot points of class k
	    plt.plot(arrayClassPoints[:,0], arrayClassPoints[:,1], 'o', markerfacecolor=tuple(col),
	             markeredgecolor='k', markersize=6)

	# plot the gaussian contour
	X, Y = np.meshgrid(np.linspace(-2, 3), np.linspace(-2,3))
	XX = np.array([X.ravel(), Y.ravel()]).T
	Z0 = score_samples(XX, mu_c[0], var_c[0])
	Z0 = Z0.reshape((50,50))
	Z1 = score_samples(XX, mu_c[1], var_c[1])
	Z1 = Z1.reshape((50,50))
	plt.contour(X, Y, Z0)
	plt.contour(X, Y, Z1)

	plt.title(title + "_" + str(epoch))
	# save figure
	saveFigureAsPNG("userEM-GMM_" + str(epoch), userFig)

def EM_GMM(dataset, n_clusters, n_features, eps, epoch = 100):
	# dataset [[x1 , x2, label], ...]

	# initialise the mu_c, sigma_c (variance), "size" pi_c
	pi_c = np.zeros(n_clusters)
	mu_c = np.zeros(shape = (n_clusters, n_features, 1))
	var_c = np.zeros(shape = (n_clusters,n_features,n_features))
	n_samples = np.zeros(n_clusters)

	# compute mean
	for data in dataset:
		if(np.int_(data[2]) == 0):
			mu_c[0,:] += np.reshape(data[0:2],(2,1))
			n_samples[0] += 1
		if(np.int_(data[2]) == 1):
			mu_c[1,:] += np.reshape(data[0:2],(2,1))
			n_samples[1] += 1

	for i in range(n_clusters):
		mu_c[i,:] /= n_samples[i]

	# compute pi_c
	for i in range(n_clusters):
		pi_c[i] = n_samples[i] / np.sum(n_samples)

	# compute covariance matrix
	for data in dataset:
		if(np.int_(data[2]) == 0):
			temp = np.reshape(data[0:2],(2,1))
			var_c[0] += np.matmul((temp - mu_c[0,:]), np.transpose(temp - mu_c[0,:]))
		if(np.int_(data[2]) == 1):
			temp = np.reshape(data[0:2],(2,1))
			var_c[1] += np.matmul((temp - mu_c[1,:]), np.transpose(temp - mu_c[1,:]))

	for i in range(n_clusters):
		var_c[i] /= n_samples[i]


	log_p_x = np.float_(0)

	for epoc in xrange(epoch):
		# Expectation Steps (E step)
		r_ic = np.zeros(shape = (len(dataset), n_clusters))

		for i, data in enumerate(dataset):
			# compute the sum used for normalisation
			normSum = np.float_(0)
			for j in xrange(n_clusters):
				normSum += (pi_c[j] * MultiGaussianProbability(np.reshape(data[0:2],(2,1)), mu_c[j,:], var_c[j]))

			for j in xrange(n_clusters):
				temp_r_ric = (pi_c[j] * MultiGaussianProbability(np.reshape(data[0:2],(2,1)), mu_c[j,:], var_c[j])) / normSum
				r_ic[i][j] = temp_r_ric[0][0]

		# update cluster labels
		for i, data in enumerate(dataset):
			data[2] = np.argmax(r_ic[i])

		# Maximization Steps (M step)

		# calculate the total responsibility allocated to cluster C
		m_c = np.zeros(n_clusters)

		for i in xrange(n_clusters):
			m_c[i] = np.sum(r_ic[:,i]) 

		# update pi_c
		for i in xrange(n_clusters):
			pi_c[i] = m_c[i] / np.sum(n_samples) # might need to update n_samples[i]

		# update the mu_c with weighted mean of assigned data
		temp_mu_c = np.zeros(shape = (n_clusters, n_features, 1))
		for i in range(n_clusters):
			for j, data in enumerate(dataset):
				temp_mu_c[i,:] += r_ic[j][i]*np.reshape(data[0:2],(2,1))
		for i in range(n_clusters):
			mu_c[i,:] = temp_mu_c[i,:]/m_c[i]

		# update covariance matrix with weighted covariance of assigned data(use new weigthed means here)
		temp_var_c = np.zeros(shape = (n_clusters,n_features,n_features))
		for i in range(n_clusters):
			for j, data in enumerate(dataset):
				temp = np.reshape(data[0:2],(2,1))
				temp_var_c[i] += r_ic[j][i]*np.matmul((temp - mu_c[i,:]),np.transpose(temp - mu_c[i,:]))

		for i in range(n_clusters):
			var_c[i] = temp_var_c[i]/m_c[i]

		# Compute log-likelihood of our model
		_log_p_x = np.float_(0)

		for i, data in enumerate(dataset):
			temp_sum = np.float(0)
			for j in xrange(n_clusters):
				temp_sum += pi_c[j] * MultiGaussianProbability(np.reshape(data[0:2], (2,1)), mu_c[j,:], var_c[j])
			_log_p_x += np.log(temp_sum)

		# # Plot the update dataset and labels
		plotDataSet(dataset, mu_c, var_c, n_clusters, epoc, 'Estimated number of clusters: 2')

		if(np.fabs(_log_p_x - log_p_x) < eps):
			log_p_x = _log_p_x
			break

		log_p_x = _log_p_x

		print "_log_p_x"
		print log_p_x


	print mu_c
	print var_c
	return {"mu_c": mu_c, "var_c": var_c}

if __name__ == "__main__":
	# # Testing MultiGaussianProbability()

	# x = np.reshape(np.array([0,0]),(2,1))
	# print x.shape
	# mu = np.reshape(np.array([0, 0]),(2,1))
	# print mu.shape
	# var = np.array([[1,0],[0,1]])
	# print var.shape
	# print MultiGaussianProbability(x,mu,var)

	# # Testing UniGaussianProbability()
	# x_1 = np.float_(4)
	# mu_1 = np.float_(4.3)
	# var_1 = np.float_(1)
	# print UniGaussianProbability(x_1,mu_1,var_1)

	GMMDataset = np.asarray(getDataset("GMM-Points.mat"))

	unique_labels = set(GMMDataset[:,2])
	colors = [plt.cm.Spectral(each)
	          for each in np.linspace(0, 1, len(unique_labels))]
	userFig0 = plt.figure(20)
	for k, col in zip(unique_labels, colors):

	    # obtain a list of points belong to class k
	    classPoints = [datapoints[0:2] for datapoints in GMMDataset if (datapoints[2] == k)]
	    # convert to numpy array which supports array[:,0] notation
	    arrayClassPoints = np.array(classPoints)
	    # plot points of class k
	    plt.plot(arrayClassPoints[:,0], arrayClassPoints[:,1], 'o', markerfacecolor=tuple(col),
	             markeredgecolor='k', markersize=6)

	plt.title('(Before) Estimated number of clusters: 2')

	# save figure
	saveFigureAsPNG("userEM-GMM_before", userFig0)

	GMM_Parameters = EM_GMM(GMMDataset, 2, 2, 0.001, 100)
	
	# plot the datapoints

	unique_labels = set(GMMDataset[:,2])
	colors = [plt.cm.Spectral(each)
	          for each in np.linspace(0, 1, len(unique_labels))]
	userFig1 = plt.figure(21)
	for k, col in zip(unique_labels, colors):

	    # obtain a list of points belong to class k
	    classPoints = [datapoints[0:2] for datapoints in GMMDataset if (datapoints[2] == k)]
	    # convert to numpy array which supports array[:,0] notation
	    arrayClassPoints = np.array(classPoints)
	    # plot points of class k
	    plt.plot(arrayClassPoints[:,0], arrayClassPoints[:,1], 'o', markerfacecolor=tuple(col),
	             markeredgecolor='k', markersize=6)

	plt.title('(After) Estimated number of clusters: 2')
	plt.show()
	# save figure
	saveFigureAsPNG("userEM-GMM_after", userFig1)

	# X, Y = np.meshgrid(np.linspace(-2, 3), np.linspace(-2,3))
	# XX = np.array([X.ravel(), Y.ravel()]).T
	# Z0 = score_samples(XX, GMM_Parameters["mu_c"][0], GMM_Parameters["var_c"][0])
	# Z0 = Z0.reshape((50,50))
	# Z1 = score_samples(XX, GMM_Parameters["mu_c"][1], GMM_Parameters["var_c"][1])
	# Z1 = Z1.reshape((50,50))
	# plt.contour(X, Y, Z0)
	# plt.contour(X, Y, Z1)
	# plt.scatter(GMMDataset[:, 0], GMMDataset[:, 1])
	 
	# plt.show()