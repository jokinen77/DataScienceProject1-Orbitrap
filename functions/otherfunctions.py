import numpy as np
import pandas as pd
from functions.modelfitting import * 

####################################
# Contains other helpful functions #
####################################

# Combines observations from multiple data frames containing identified peaks into a single data frame
def combineObservations(peaklists):
	peaklist = peaklists[0][["formula", "mz", "observed"]].set_index("formula")
	for i in range(len(peaklists)-1):
		peaklist = peaklist.join(peaklists[i+1][["formula", "observed"]].set_index("formula"), on="formula", rsuffix='_'+str(i+1))
	return peaklist

# Combines multiple data frames containing identified peaks  into a single data frame
def combinePeaklists(peaklists):
	peaklist = peaklists[0].set_index("formula")
	for i in range(len(peaklists)-1):
		peaklist = peaklist.join(peaklists[i+1].drop("mz", axis=1).set_index("formula"), on="formula", rsuffix='_'+str(i+1))
	return peaklist
	
# Calculate errors for one selection method
# with different values of n and k, where
# 6 <= n*k <= maxAmount
# n < amount, k < amount
# mainly used for plotting these errors for data analysis
def calculateErrors(peaklists, selection, amount, maxAmount):
	errors = np.zeros([5, len(peaklists), amount, amount])
	for degree in range(5):
		for pl in range(len(peaklists)):
			ind = np.linspace(0,len(peaklists[pl])-1,len(peaklists[pl]))
			test = peaklists[pl].iloc[ind%5==0,:]
			train = peaklists[pl].iloc[ind%5!=0,:]
			for i in range(amount):
				for ii in range(amount):
					if i*ii>=6 and i*ii<=maxAmount:
						errors[degree,pl,i,ii] = fitPolynomial(train, degree+1, selection, i, ii, 0.5, False, False, test)[1]
	return errors