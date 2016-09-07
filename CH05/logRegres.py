import numpy as np
import pandas as pd
import random
import math


# A function to load data
def loadData():
	data = pd.read_table('testSet.txt').values
	(m, n) = data.shape
	features = np.ones((m, n))
	features[:, 1:] = data[:,:-1]
	labels = data[:, -1]
	return features, labels

#  A sigmoid function implementation
def sigmoid(input):
	return 1.0 / (1 + np.exp(-1 * input))

# Batch gradient ascent
def gradAscent(dataSet, labels, alpha=0.001, numOfIter=500):
	dataSet, labels = np.array(dataSet), np.array(labels)
	(m, n) = dataSet.shape
	labels = labels.reshape(m, 1)
	weights = np.ones((n, 1))
	for i in range(numOfIter):
		predictions = sigmoid(np.dot(dataSet, weights))
		error = labels - predictions
		weights += alpha * np.dot(dataSet.transpose(), error)
	return weights

# Stachastic gradient ascent
def stocGradAscent(dataSet, labels, alpha=0.001, numOfIter=200):
	dataSet, labels = np.array(dataSet), np.array(labels)
	(m, n) = dataSet.shape
	weights = np.ones(n)
	for i in range(numOfIter):
		rand_seq = random.sample(list(np.arange(m)), m)
		for j in rand_seq:
			alpha = 4/(1.0+j+i)+0.0001
			chosen_example = dataSet[j]
			label = labels[j]
			predictions = sigmoid(sum(chosen_example * weights))
			error = label - predictions
			weights += alpha * error * chosen_example
	return weights


# Classify a new vector
def classifyVector(input, weights):
	prob = sigmoid(sum(input * weights))
	return prob >= 0.5

def colicTest():
	training_set, training_labels = [], []
	testing_set, testing_labels = [], []
	with open('horseColicTraining.txt') as training_data:
		for line in training_data.readlines():
			curr_line = line.strip().split()
			training_set.append([float(item) for item in curr_line[:-1]])
			training_labels.append(float(curr_line[-1]))
	with open('horseColicTest.txt') as testing_data:
		for line in testing_data.readlines():
			curr_line = line.strip().split()
			testing_set.append([float(item) for item in curr_line[:-1]])
			testing_labels.append(float(curr_line[-1]))
	weights = stocGradAscent(training_set, training_labels, 0.001, 1000)
	error = 0.0
	for i in range(len(testing_set)):
		prediction = classifyVector(testing_set[i], weights)
		if prediction != testing_labels[i]:
			error += 1
	return error / len(testing_set)

def multiTest():
	repeating = 10
	cumul_error = 0.0
	for i in range(repeating):
		cumul_error += colicTest()
	return cumul_error / repeating



