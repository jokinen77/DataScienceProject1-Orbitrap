import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from functions.peakutils import * 
from functions.modelfitting import * 

############################################################
# Contains functions for plotting different kinds of data. #
############################################################
	
# Plot a peak i from a peaklist that's fram data
def plotPeak(i, peaklist, data):
	if i < len(peaklist.iloc[:,0]):
		peakindex = findPeakIndices(data)
		ii = int(peaklist.iloc[i,5])
		peak = data[peakindex[ii,0]:peakindex[ii,1]+1]
		plt.plot(peak[:,0],peak[:,1])
		middle = round(len(peak[:,0])/2)
		x = np.linspace(peak[0,0],peak[-1,0],1000)
		y = peaklist.iloc[i,3]*np.exp(-0.5*((x-peaklist.iloc[i,2])/peaklist.iloc[i,4])**2)
		plt.plot(x,y)
		plt.vlines(peaklist.iloc[i,2],0,max(peak[:,1]))
		plt.vlines(peaklist.iloc[i,1],0,max(peak[:,1]), color = "green")
		plt.legend(["Data","Fitted normal distribution","Mean from data","True mean"])
		plt.title(peaklist.iloc[i,0])
		plt.show()
		

# Plot peaklist with changing n or k peak selection parameters using a polynomial fit
def plotPeaklists(peaklists, degree, change_n, start_n, start_k, amount, onlyMean=True):
	xlab = "n" if change_n else "k"
	ylab = "log MSE"
	title = "n best from " + str(start_k) + " partitions - " if change_n else str(start_n) + " best from k partitions - "
	title = "degree = " + str(degree) + " - " + title

	errors = np.zeros([len(peaklists),amount,4])
	if change_n:
		for pl in range(len(peaklists)):
			ind = np.linspace(0,len(peaklists[pl])-1,len(peaklists[pl]))
			test = peaklists[pl].iloc[ind%5==0,:]
			train = peaklists[pl].iloc[ind%5!=0,:]
			for i in range(amount):
				errors[pl,i,:] = fitPolynomialAll(train, degree, start_n + i, start_k, 0.5, False, False, test_set=test)[2]
	else: 
		for pl in range(len(peaklists)):
			ind = np.linspace(0,len(peaklists[pl])-1,len(peaklists[pl]))
			test = peaklists[pl].iloc[ind%5==0,:]
			train = peaklists[pl].iloc[ind%5!=0,:]
			for i in range(amount):
				errors[pl,i,:] = fitPolynomialAll(train, degree, start_n, start_k + i, 0.5, False, False, test_set=test)[2]
	
	
	if change_n:
		x = np.linspace(start_n,start_n+amount-1,amount)
	else:
		x = np.linspace(start_k,start_k+amount-1,amount)
		
	lims = [(np.quantile(np.log(errors[:,:,i]), 0.025), np.max([np.quantile(np.log(errors[:,:,i]), 0.975), 0.2])) for i in range(4)]
	
	fig, axs = plt.subplots(2, 2)
	
	if not onlyMean:	
		for pl in range(len(peaklists)):
			axs[0, 0].plot(x,np.log(errors[pl,:,0]), color="green", alpha=0.5)
	axs[0, 0].plot(x,[np.mean(np.log(errors[:,y,0])) for y in range(amount)], color="black", linewidth=3)
	axs[0, 0].plot(x,[np.quantile(np.log(errors[:,y,0]), 0.05) for y in range(amount)], color="red", linewidth=2)
	axs[0, 0].plot(x,[np.quantile(np.log(errors[:,y,0]), 0.95) for y in range(amount)], color="red", linewidth=2)
	axs[0, 0].plot(x,np.zeros(amount), ':', color="black", linewidth=3)
	axs[0, 0].set_title(title+"median error")
	axs[0, 0].legend(["mean", "90% quantile"])
	axs[0, 0].set_ylabel(ylab)
	axs[0, 0].set_xlabel(xlab)
	axs[0, 0].set_ylim(lims[0])
	
	if not onlyMean:
		for pl in range(len(peaklists)):
			axs[0, 1].plot(x,np.log(errors[pl,:,1]), color="green", alpha=0.5)
	axs[0, 1].plot(x,[np.mean(np.log(errors[:,y,1])) for y in range(amount)], color="black", linewidth=3)
	axs[0, 1].plot(x,[np.quantile(np.log(errors[:,y,1]), 0.05) for y in range(amount)], color="red", linewidth=2)
	axs[0, 1].plot(x,[np.quantile(np.log(errors[:,y,1]), 0.95) for y in range(amount)], color="red", linewidth=2)
	axs[0, 1].plot(x,np.zeros(amount), ':', color="black", linewidth=3)
	axs[0, 1].set_title(title+"minimum error")
	axs[0, 1].legend(["mean", "90% quantile"])
	axs[0, 1].set_ylabel(ylab)
	axs[0, 1].set_xlabel(xlab)
	axs[0, 1].set_ylim(lims[1])

	if not onlyMean:
		for pl in range(len(peaklists)):
			axs[1, 0].plot(x,np.log(errors[pl,:,2]), color="green", alpha=0.5)
	axs[1, 0].plot(x,[np.mean(np.log(errors[:,y,2])) for y in range(amount)], color="black", linewidth=3)
	axs[1, 0].plot(x,[np.quantile(np.log(errors[:,y,2]), 0.05) for y in range(amount)], color="red", linewidth=2)
	axs[1, 0].plot(x,[np.quantile(np.log(errors[:,y,2]), 0.95) for y in range(amount)], color="red", linewidth=2)
	axs[1, 0].plot(x,np.zeros(amount), ':', color="black", linewidth=3)
	axs[1, 0].set_title(title+"minimum + median error")
	axs[1, 0].legend(["mean", "90% quantile"])
	axs[1, 0].set_ylabel(ylab)
	axs[1, 0].set_xlabel(xlab)
	axs[1, 0].set_ylim(lims[2])
	
	if not onlyMean:
		for pl in range(len(peaklists)):
			axs[1, 1].plot(x,np.log(errors[pl,:,3]), color="green", alpha=0.5)
	axs[1, 1].plot(x,[np.mean(np.log(errors[:,y,3])) for y in range(amount)], color="black", linewidth=3)
	axs[1, 1].plot(x,[np.quantile(np.log(errors[:,y,3]), 0.05) for y in range(amount)], color="red", linewidth=2)
	axs[1, 1].plot(x,[np.quantile(np.log(errors[:,y,3]), 0.95) for y in range(amount)], color="red", linewidth=2)
	axs[1, 1].plot(x,np.zeros(amount), ':', color="black", linewidth=3)
	axs[1, 1].set_title(title+"highest intensity")
	axs[1, 1].legend(["mean", "90% quantile"])
	axs[1, 1].set_ylabel(ylab)
	axs[1, 1].set_xlabel(xlab)
	axs[1, 1].set_ylim(lims[3])
	
	
	
# plot peak lists based on pre-calculated errors using 
# calculateErrors from otherfunctions
# vmin and vmax provide the minimum and maximum values
# for the resulting heatmap.
# The heatmap is drawn based on logarithm of the errors
def plotPeaklists3(errors, title="", vmin=-2, vmax=2):
	plt.rcParams['figure.figsize'] = [16, 4]
	sns.set()
	fig, ax = plt.subplots(1, 5)
	fig.suptitle(title)
	amount = len(errors[0,0,0,:])
	for degree in range(5):
		avg = np.zeros([amount, amount])
		for i in range(amount):
			for ii in range(amount):
				avg[i,ii]=np.log(np.mean(errors[degree,:,i,ii]))
		sns.heatmap(avg, ax=ax[degree], square=True, cbar=False, xticklabels=4, yticklabels=4, vmin=vmin, vmax=vmax, cmap="coolwarm")
		ax[degree].set_title("Degree " + str(degree+1))
		ax[degree].set_xlabel("k")
		ax[degree].set_ylabel("n")
		ax[degree].tick_params(left=True, bottom=True, right=False, top=False)