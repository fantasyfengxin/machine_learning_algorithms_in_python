import numpy as np
import pickle
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

# Find the mojority label from a list of labels
def majorityCnt(labels):
	label_count = defaultdict(int)
	maximum_label = ('', 0)
	for label in labels:
		label_count[label] += 1
		if label_count[label] > maximum_label[1]:
			maximum_label = (label, label_count[label])
	return maximum_label[0]

def splitData(dataSet, index, value):
	dataSet = np.array(dataSet)
	sub_dataSet = dataSet[dataSet[:, index] == value]
	sub_dataSet = np.delete(sub_dataSet, index, axis=1)
	return sub_dataSet

def choose_split_feature(dataSet):
	dataSet = np.array(dataSet)
	current_entropy = calcShannonEnt(dataSet)
	max_information_gain = 0.0
	split_feature = -1
	# Iterate through all features
	for i in range(len(dataSet[0]) - 1):
		feature_values = set(dataSet[:, i])
		temp_entropy = 0.0
		for value in feature_values:
			sub_dataSet = splitData(dataSet, i, value)
			weight = float(len(sub_dataSet)) / len(dataSet)
			temp_entropy += weight * calcShannonEnt(sub_dataSet)
		information_gain = current_entropy - temp_entropy
		if information_gain > max_information_gain:
			max_information_gain = information_gain
			split_feature = i
	return split_feature

def createTree(dataSet, label_names):
	dataSet = np.array(dataSet)
	labels = np.array([row[-1] for row in dataSet])
	if len(set(labels)) == 1: # all examples have the same label
		return labels[0]
	if len(dataSet[0]) == 1: # no feature to split
		return majorityCnt(labels)
	# get the index of best feature to split
	split_feature = choose_split_feature(dataSet)
	split_feature_label = label_names[split_feature]
	my_tree = {split_feature_label: {}}
	split_feature_values = set(dataSet[:, split_feature])
	# update label_names for recursion
	del(label_names[split_feature])
	for value in split_feature_values:
		sub_label_names = label_names[:]
		sub_dataSet = splitData(dataSet, split_feature, value)
		my_tree[split_feature_label][value] = createTree(sub_dataSet, sub_label_names)
	return my_tree

def classify(input_tree, label_names, test_array):
	test_array = np.array(test_array)
	tree_feature_name = list(input_tree.keys())[0]
	tree_feature_dict = input_tree[tree_feature_name]
	test_feature_value = str(test_array[label_names.index(tree_feature_name)])
	tree_feature_value = tree_feature_dict[test_feature_value]
	if isinstance(tree_feature_value, dict):
		label = classify(tree_feature_value, label_names,test_array)
	else: label = tree_feature_value
	return label

def storeTree(input_tree, filename):
	with open(filename, 'wb') as file:
		pickle.dump(input_tree, file)

def grabTree(filename):
	with open(filename, 'rb') as file:
		return pickle.load(file)
