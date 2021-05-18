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

'''
This function selects a set amount n of the best peaks to lead the calibration process, where best means the minum error (abs(detected_mz-observed_mz)).
It is possible to set a treshold to specifiy the minimal value of error accepted.
The threshold criteria is secondary to the number of peaks, meaning that if the threshold isn't met and the number of peaks selected is under n, it will put those below the threshold to reach n. To change this set the forceThreshold parameter to True.
To disable the threshold set it to 0 (default value).
The treshold is treated as absolute value, so negative values don't matter.
This method provides the remaining peaks as a test set, and the size can be specified with the test_n parameter. If left at 0 it will return all the discarded data for testing.
The randomTestSamples parameter allows to sample randomly from the remaining data, otherwise they're provided ordered by least uncerteinty value.
'''
def selectBestPeaks(data, n=10, threshold=0, forceThreshold=False, test_n=0, randomTestSamples=True, debug=False):
	if n <=0:
		return None
	result = data.copy()
	result["absolute_error"] =  abs(data["error"])
	result = result.sort_values(by=['absolute_error'], ascending=True) #specified ascending parameter for flexibility
	diff = result
	test_n = abs(test_n)

	if debug:
		print(result)

	if threshold:
		threshold = abs(threshold)
		filtered = result[result['absolute_error']<=threshold]

	if len(result) > n:
		if threshold:
			if len(filtered) > n:
				result = filtered[0:n]
			else:
				if forceThreshold:
					result = filtered
				else:
					result = result[0:n]
		else:
			result = result[0:n]
	else:
		if forceThreshold and threshold:
			result = filtered
		else:
			x = max(len(result), n)
			result = result[0:x]

	diff = pd.concat([result,diff]).drop_duplicates(keep=False)
	if test_n != 0:
		test_n = min(test_n, len(diff))
		if randomTestSamples:
			diff = diff.sample(test_n)
		else:
			diff = diff[0:test_n]
	return result, diff
