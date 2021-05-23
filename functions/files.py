import numpy as np
import pandas as pd
from os import listdir
from os.path import isfile, join

####################################################################
# Contains functions for reading files, saving data to files, etc. #
####################################################################

# Read a file in the correct format for these functions
def readFile(fileName):
	return pd.read_csv(fileName).iloc[3:,:2].to_numpy().astype("float64")
	

# Process all files from path
# Can take a long time for a large amount of files
# pl is a csv file containing known peaks
def getPeaklists(path, th=1e-4, pl="peaklist.csv"):
	files = [f for f in listdir(path) if isfile(join(path, f))]
	plist = pd.read_csv(pl)
	peaklists = [None]*len(files)
	for i in range(len(files)):
		peaklists[i] = identifyPeaks(readFile(join(path, files[i])), plist, th)
		if (i+1)%10 == 0:
			print(str(i+1) + "/" + str(len(files)) + " files processed.")

	return peaklists
	
# saves peaklists to a specified path 
def savePeaklists(peaklists, path, name="peaklist"):
	for i in range(len(peaklists)):
		peaklists[i].to_csv(join(path, name + str(i) + ".csv"))

# loads peaklists from a specified path
def loadPeaklists(path):
	files = [f for f in listdir(path) if isfile(join(path, f))]
	peaklists = [None]*len(files)
	for i in range(len(files)):
		peaklists[i] = pd.read_csv(join(path, files[i]), index_col=0)
		if (i+1)%10 == 0:
			print(str(i+1) + "/" + str(len(files)) + " files loaded.")
	return peaklists