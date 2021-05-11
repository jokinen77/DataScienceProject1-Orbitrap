from scipy.optimize import curve_fit
import numpy as np
from scipy.optimize import curve_fit
from functions.peakselection import * 
import matplotlib.pyplot as plt

# Polynomial fit using one selection method, returns parameters for the model and the error
# 
# pl is a pandas data frame containing identified peakselection
# degree of the polynomial has to be 1-5
# selection is the peakselection method used
# n is the number of peaks chosen from each partition
# k is the number of partitions
# n*k cannot exceed the amount of peaks in pl!
# if plot is True, the results are plotted
# if printErrors is True, the resulting errors are printed
# test_set to calculate the errors from. If None, the peaks not selected for fitting the function are used as a test set.
def fitPolynomial(pl, degree=3, selection=selectPeaksHighestIntensity, n=3, k=5, q=0.6, plot=True, printErrors=True, test_set=None):
	train, test = selectPeaks(pl, n, k, selection)
	
	if test is not None:
		test = test_set
	
	train_x = train["observed"]
	train_y = train["mz"]
	test_x = test["observed"]
	test_y = test["mz"]

	functions = [lambda x, a: a, 
		lambda x, a, b: a + b*x, 
		lambda x, a, b, c: a + b*x + c*x**2, 
		lambda x, a, b, c, d: a + b*x + c*x**2 + d*x**3, 
		lambda x, a, b, c, d, e: a + b*x + c*x**2 + d*x**3 + e*x**4]
	f = functions[degree-1]

	p = curve_fit(f, train_x, train_x-train_y)[0]

	if 5 - degree > 0:
		p0 = np.zeros(5)
		p0[0:degree] = p
		p = p0

	f = functions[4]
	fit = f(test_x, p[0], p[1], p[2], p[3], p[4])

	if plot:
		plt.scatter(train_x, train_x-train_y)
		plt.scatter(test_x, test_x-test_y)
		x_coor = np.linspace(min(test_x), max(test_x), 1000)
		plt.plot(x_coor, f(x_coor, p[0], p[1], p[2], p[3], p[4]))
		plt.legend(["training data", "test data", "model"])
		plt.show()
	
	err = sum((test_x-test_y - fit)**2)/sum((test_x-test_y)**2)

	if printErrors:
		print(str(round(err*10000)/100)+"% of the prefit MSE")

	
	return p, err

# Polynomial fit using a combination of 2 peak selection methods
# q*f_selection1(x) + (1-q)*f_selection2(x)
# Returns the parameters for both functions with the error
# see fitPolynomial for information about parameters 
def fitPolynomial2(pl, degree=3, selection1=selectPeaksMinimumError, selection2=selectPeaksMedianError, n=3, k=5, q=0.5, plot=True, printErrors=True, test_set=None):
	train, test = selectPeaks(pl, n, k, selection1)
	train2, test2 = selectPeaks(pl, n, k, selection2)
	
	test = pl[pl["mz"].isin(set(test["mz"]).intersection(set(test2["mz"])))]
	
	if test is not None:
		test = test_set
	
	train_x = train["observed"]
	train_y = train["mz"]
	train2_x = train2["observed"]
	train2_y = train2["mz"]
	test_x = test["observed"]
	test_y = test["mz"]

	functions = [lambda x, a: a, 
		lambda x, a, b: a + b*x, 
		lambda x, a, b, c: a + b*x + c*x**2, 
		lambda x, a, b, c, d: a + b*x + c*x**2 + d*x**3, 
		lambda x, a, b, c, d, e: a + b*x + c*x**2 + d*x**3 + e*x**4]
	f = functions[degree-1]

	p = curve_fit(f, train_x, train_x-train_y)[0]
	p2 = curve_fit(f, train2_x, train2_x-train2_y)[0]

	if 5 - degree > 0:
		p0 = np.zeros(5)
		p0[0:degree] = p
		p = p0
		p0 = np.zeros(5)
		p0[0:degree] = p2
		p2 = p0

	f = functions[4]
	fit = (q*f(test_x, p[0], p[1], p[2], p[3], p[4]) + (1.0-q)*f(test_x, p2[0], p2[1], p2[2], p2[3], p2[4]))

	if plot:
		plt.scatter(test_x, test_x-test_y)
		x_coor = np.linspace(min(test_x), max(test_x), 1000)
		plt.plot(x_coor, (q*f(x_coor, p[0], p[1], p[2], p[3], p[4]) + (1.0-q)*f(x_coor, p2[0], p2[1], p2[2], p2[3], p2[4])))
		plt.show()
	
	err = np.mean((test_x-test_y - fit)**2)/np.mean((test_x-test_y)**2)

	if printErrors:
		print(str(round(err*10000)/100)+"% of the prefit MSE")
	
	return p, p2, err


# Polynomial fit using 4 different peak selection methods
# Mainly used for comparing these methods
# Degree has to be 1-5
def fitPolynomialAll(pl, degree=3, n=3, k=5, q=0.6, plot=True, printErrors=True, test_set=None):
	train, test = selectPeaks(pl, n, k, selectPeaksMedianError)
	train2, test2 = selectPeaks(pl, n, k, selectPeaksMinimumError)
	train3, test3 = selectPeaks(pl, n, k, selectPeaksHighestIntensity)
	
	test = pl[pl["mz"].isin(set(test["mz"]).intersection(set(test2["mz"])))]
	
	if test_set is not None:
		test = test_set
		test2 = test_set
		test3 = test_set
	
	train_x = train["observed"]
	train_y = train["mz"]
	train2_x = train2["observed"]
	train2_y = train2["mz"]
	train3_x = train3["observed"]
	train3_y = train3["mz"]
	test_x = test["observed"]
	test_y = test["mz"]
	test3_x = test3["observed"]
	test3_y = test3["mz"]

	functions = [lambda x, a: a, 
		lambda x, a, b: a + b*x, 
		lambda x, a, b, c: a + b*x + c*x**2, 
		lambda x, a, b, c, d: a + b*x + c*x**2 + d*x**3, 
		lambda x, a, b, c, d, e: a + b*x + c*x**2 + d*x**3 + e*x**4]
	f = functions[degree-1]

	p = curve_fit(f, train_x, train_x-train_y)[0]
	p2 = curve_fit(f, train2_x, train2_x-train2_y)[0]
	p3 = curve_fit(f, train3_x, train3_x-train3_y)[0]

	if 5 - degree > 0:
		p0 = np.zeros(5)
		p0[0:degree] = p
		p = p0
		p0 = np.zeros(5)
		p0[0:degree] = p2
		p2 = p0
		p0 = np.zeros(5)
		p0[0:degree] = p3
		p3 = p0
	f = functions[4]
	fit = f(test_x, p[0], p[1], p[2], p[3], p[4])
	fit2 = (q*f(test_x, p[0], p[1], p[2], p[3], p[4]) + (1.0-q)*f(test_x, p2[0], p2[1], p2[2], p2[3], p2[4]))
	fit3 = f(test_x, p2[0], p2[1], p2[2], p2[3], p2[4])
	fit4 = f(test3_x, p3[0], p3[1], p3[2], p3[3], p3[4])

	if plot:
		plt.scatter(test_x, test_x-test_y)
		x_coor = np.linspace(min(test_x), max(test_x), 1000)
		plt.plot(x_coor, f(x_coor, p[0], p[1], p[2], p[3], p[4]))
		plt.plot(x_coor, f(x_coor, p2[0], p2[1], p2[2], p2[3], p2[4]))
		plt.plot(x_coor, (q*f(x_coor, p[0], p[1], p[2], p[3], p[4]) + (1.0-q)*f(x_coor, p2[0], p2[1], p2[2], p2[3], p2[4])))
		plt.plot(x_coor, f(x_coor, p3[0], p3[1], p3[2], p3[3], p3[4]))
		plt.legend(["Median error", "Minimum error", "Median + Minimum error", "Highest intensity"])
		plt.show()
	
	err1 = np.mean((test_x-test_y - fit)**2)/np.mean((test_x-test_y)**2)
	err2 = np.mean((test_x-test_y - fit3)**2)/np.mean((test_x-test_y)**2)
	err3 = np.mean((test_x-test_y - fit2)**2)/np.mean((test_x-test_y)**2)
	err4 = np.mean((test3_x-test3_y - fit4)**2)/np.mean((test3_x-test3_y)**2)
	if printErrors:
		print("Median error:")
		print(str(round(err1*10000)/100)+"% of the prefit MSE")
		print()
		print("Minimum error:")
		print(str(round(err2*10000)/100)+"% of the prefit MSE")
		print()
		print("Median + Minimum error:")
		print(str(round(err3*10000)/100)+"% of the prefit MSE")
		print()
		print("Highest intensity:")
		print(str(round(err4*10000)/100)+"% of the prefit MSE")
	
	return p, p2, np.array([err1, err2, err3, err4])
	
