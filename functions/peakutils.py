import numpy as np
from scipy.optimize import minimize
import scipy.signal as scs
import pandas as pd

##############################################################
# Contains functions for finding peaks and identifying them. #
##############################################################


# Find indices of the different peaks in the data
def findPeakIndices(data):
	n = int(round(sum(data[:,1] == 0)/2)+1)
	indices = np.zeros([n,2], dtype="uint32")
	a = False
	ii = 0
	for i in range(len(data[:,1])-1):
		if data[i,1] == 0 and a:
			indices[ii, 1] = i - 1
			if ii != n:
				indices[ii+1, 0] = i
			ii += 1
		a = data[i,1] == 0
	last = n-sum(indices[:,0]>=indices[:,1])
	indices = indices[0:last+1,:]
	indices[last, 1] = len(data[:,0]) - 1
	return indices
	

# Calculate mean square error for the model
def MSE(x, y, a, mu):
	norm = lambda a, mu, x: a*np.exp(-0.5*((x-mu)**2/(mu**2/(4*280000**2))))
	error = 0
	for i in range(len(y)):
		error = error + (y[i] - np.sum(norm(a,mu,x[i])))**2
	return error/len(y)

# Returns a function that is used in getMeans
def errorFunction(x, y):
	return lambda params: MSE(x, y, params[::2], params[1::2])

# Function used to group values that are too close (abs(mu[n]-mu[n+1])<th) to each other together
def group(mu, th):
	groups = []
	unassigned = np.linspace(0,len(mu)-1,len(mu)).astype(int)
	while len(unassigned) > 0:
		dist = abs(np.array(mu[unassigned]) - mu[unassigned[0]])
		group = unassigned[dist < th]
		unassigned = unassigned[dist >= th]
		groups += [group]
	return groups



# Get one or more means depending on if there are multiple peaks in the data
# by fitting one or more Gaussian distributions to the data using resolution of 280000 
def getMeans(peaks):
	ind = scs.find_peaks(peaks[:,1], max(peaks[:,1])/20, prominence=max(peaks[:,1])/10)[0]
	mu = peaks[ind,0]
	n = len(mu)
	
	init  = []
	for i in range(len(mu)):
		init += [np.mean(peaks[:,1]), mu[i]]
	
	# Fit n Gaussian distributions to the data
	f = errorFunction(peaks[:,0], peaks[:,1])
	params = minimize(f, init, method='BFGS').x
	
	a = params[::2]
	mu = params[1::2]
	sigma = mu/(280000*2*np.sqrt(2*np.log(2)))
	
	# Group fitted distributions that are too close to each other together
	# and combine them
	groups = group(mu, 1e-5)
	a2 = np.zeros(len(groups))
	mu2 = np.zeros(len(groups))
	sigma2 = np.zeros(len(groups))
	for i in range(len(groups)):
		mu2[i] = np.mean(mu[groups[i]])
		a2[i] = np.sum(a[groups[i]])
		sigma2[i] = np.mean(sigma[groups[i]])
	return mu2, a2, sigma2



# Get all peak means for the data using getMeans function
def getAllMeans(peaks):
	indices = findPeakIndices(peaks)
	n = len(indices[:,0])
	means = []
	aa = []
	sigmas = []
	peaki = []
	for i in range(n):
		peak = peaks[indices[i,0]:indices[i,1]+1,:]
		mu, a, sigma = getMeans(peak)
		for ii in range(len(a)):
			means += [mu[ii]]
			aa += [a[ii]]
			sigmas += [sigma[ii]]
			peaki += [i]
	return means, aa, sigmas, peaki
	
	
# Identify peaks and return a data frame of peaks and their properties
# uses getAllMeans to do so
def identifyPeaks(data, peaklist, th):
	peaklist['observed'] = np.zeros(len(peaklist['mz']))
	peaklist['a'] = np.zeros(len(peaklist['mz']))
	peaklist['sigma'] = np.zeros(len(peaklist['mz']))
	peaklist['peak'] = np.zeros(len(peaklist['mz']))
	peakMeans, a, sigmas, ind = getAllMeans(data)
	trueValue = peaklist['mz'].to_numpy()
	for i in range(len(trueValue)):
		if min(abs(peakMeans - trueValue[i])) < th:
			j = np.argmin(abs(peakMeans - trueValue[i]))
			peaklist.iloc[i,2] = peakMeans[j]
			peaklist.iloc[i,3] = a[j]
			peaklist.iloc[i,4] = sigmas[j]
			peaklist.iloc[i,5] = ind[j]
	unidentified = np.linspace(0, len(trueValue)-1, len(trueValue))
	unidentified = unidentified[peaklist['observed'].to_numpy() == 0]
	return peaklist.drop(unidentified)
	
	

