import numpy as np
from math import log
from collections import defaultdict

def createDataSet():
	dataSet = [ [1, 1, 'yes'],
				[1, 1, 'yes'],
				[1, 0, 'no'],
				[0, 1, 'no'],
				[0, 1, 'no']]
	dataSet = np.array(dataSet)
	labels = ['no surfacing', 'flippers']
	return dataSet, labels


def calcShannonEnt(dataSet):
	dataSet = np.array(dataSet)
	data_size = len(dataSet)
	label_counts = defaultdict(int)
	for item in dataSet:
		label = item[-1]
		label_counts[label] += 1
	entropy = 0.0
	for key in label_counts.keys():
		prob = label_counts[key] / float(data_size)
		entropy -= prob * log(prob, 2)
	return entropy 
