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


def createTree(dataSet, label_names):
	labels = np.array([row[-1] for row in dataSet])
	if len(set(labels)) == 1:
		return labels[0]
	if len(dataSet[0]) == 1:
		return majorityCnt(labels)
	best_feature = choose_best_feature(dataSet)
	best_feature_label = label_names[best_feature]
	my_tree = {best_feature_label: {}}
	best_feature_values = dataSet[:, best_feature]
	best_feature_values = set(best_feature_values)
	label_names = del label_names[best_feature]
	for value in best_feature_values:
		sub_label_names = label_names[:]
		my_tree[best_feature_label][value] = createTree(splitData(dataSet, best_feature, value), sub_label_names)
	return my_tree

