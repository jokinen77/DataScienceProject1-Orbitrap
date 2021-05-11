import pandas as pd
import numpy as np

# Choose n peaks with the smallest error
def selectPeaksMinimumError(peaks, n):
	errors = abs(peaks.iloc[:,1]-peaks.iloc[:,2])
	ind = np.argpartition(errors, n)
	return peaks.iloc[ind[:n],:], peaks.iloc[ind[n:],:]

# Choose n peaks with error closest to the mean error
def selectPeaksMeanError(peaks, n):
	errors = abs(peaks.iloc[:,1]-peaks.iloc[:,2])
	ind = np.argpartition(abs(errors-np.mean(errors)), n)
	return peaks.iloc[ind[:n],:], peaks.iloc[ind[n:],:]

# Choose n peaks with error closest to the median error
def selectPeaksMedianError(peaks, n):
	errors = abs(peaks.iloc[:,1]-peaks.iloc[:,2])
	ind = np.argpartition(abs(errors-np.median(errors)), n)
	return peaks.iloc[ind[:n],:], peaks.iloc[ind[n:],:]

# Calculate intensity from a and sigma values that you can find from the identifyPeaks() data  frame
def getIntensity(a, sigma):
	return a/sigma*np.sqrt(2*np.pi)

# Calculate logarithm of intensity from a and sigma values that you can find from the identifyPeaks() data  frame
def getLogIntensity(a, sigma):
	return np.log(a) - np.log(sigma) + 0.5*np.log(2*np.pi)

# Choose n peaks with the highest intensity
def selectPeaksHighestIntensity(peaks, n):
	intensity = getLogIntensity(peaks.iloc[:, 3], peaks.iloc[:, 4])
	ind = np.argpartition(-1*intensity, n)
	return peaks.iloc[ind[:n],:], peaks.iloc[ind[n:],:]


# Splits the peaks into k partitions and selects n peaks from each partition
# so it results in n*k peaks for training and the rest of the peaks for testing
def selectPeaks(peaklist, n, k, select=selectPeaksMinimumError):
	n_peaks = len(peaklist.iloc[:,0])
	partition_size = int((n_peaks - (n_peaks % k)) / k)
	test = pd.DataFrame(columns=["formula","mz","observed","a","sigma","peak"])
	train = pd.DataFrame(columns=["formula","mz","observed","a","sigma","peak"])
	for i in range(k):
		if i < k-1:
			partition = peaklist.iloc[i*partition_size:(i+1)*partition_size,:]
		else:
			partition = peaklist.iloc[i*partition_size:,:]
		tr, ts = select(partition, n)
		test = test.append(ts)
		train = train.append(tr)
	return train, test