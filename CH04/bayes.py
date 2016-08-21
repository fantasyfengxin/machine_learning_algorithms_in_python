import numpy as np
from math import log

def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]    #1 is abusive, 0 not
    return postingList,classVec


# A function to generate vocabulary list
def createVocaList(data_set):
	vocal_list = set()
	for document in data_set:
		vocal_list = vocal_list | set(document)
	return list(vocal_list)

# A function to generate training matrix
def setOfWords2Vec(vocal_list, input_set):
	training_matrix = []
	for document in input_set:
		example = [0] * len(vocal_list)
		for word in document:
			if word in vocal_list:
				example[vocal_list.index(word)] = 1
		training_matrix.append(example)
	return np.array(training_matrix)

# Training function
def trainNB0(training_matrix, training_labels):
	training_matrix, training_labels = np.array(training_matrix), np.array(training_labels)
	training_size = len(training_labels)
	feature_size = len(training_matrix[0])
	unique_labels = list(set(training_labels))
	feature_prob = {}
	words_count = {}
	label_prob = {}
	for label in unique_labels:
		feature_prob[label] = np.ones(feature_size)
		label_prob[label] = 0
		words_count[label] = feature_size
	for i in range(training_size):
		label = training_labels[i]
		vector = training_matrix[i]
		label_prob[label] += 1
		feature_prob[label] += vector 
		words_count[label] += sum(vector)
	for key in unique_labels:
		feature_prob[key] = np.array([ log(val) for val in feature_prob[key] / words_count[key] ]) 
		label_prob[key] = log(label_prob[key] / training_size)
	return feature_prob, label_prob

# Testing function
def classifyNB(vec_to_classify, feature_prob, label_prob):
	vec_to_classify = np.array(vec_to_classify)
	unique_labels = list(set([label for label in label_prob.keys()]))
	max_prob, res_lable = -float('inf'), 0
	for label in unique_labels:
		conditional_prob = sum(vec_to_classify * feature_prob[label])
		final_prob = conditional_prob + label_prob[label]
		if final_prob > max_prob:
			max_prob = final_prob
			res_label = label
	return res_label

# A wrapper function for loading data, training, and testing
def testingNB():
	documents, labels = loadDataSet()
	vocal_list = createVocaList(documents)
	training_matrix = setOfWords2Vec(vocal_list, documents)
	feature_prob, label_prob = trainNB0(training_matrix, labels)
	test_entry = [['stupid', 'garbage']]
	entry_repr = setOfWords2Vec(vocal_list, test_entry)[0]
	print(test_entry, 'classified as: ', classifyNB(entry_repr, feature_prob, label_prob))

# A function to parse the document
def textParse(bigString):
	import re
	list_of_tokens = re.split(r'\W*', bigString)
	return [tok.lower() for tok in list_of_tokens if len(tok) > 2]

# A function to do spam email classification
def spamTest():
	import random
	doc_list = []; labels = [];
	for i in range(1, 26):
		with open('email/spam/{0}.txt'.format(i)) as file:
			word_list = textParse(file.read())
			doc_list.append(word_list)
			labels.append(1)
		with open('email/ham/{0}.txt'.format(i)) as file:
			word_list = textParse(file.read())
			doc_list.append(word_list)
			labels.append(0)
	vocal_list = createVocaList(doc_list)
	training_set, testing_set = [], []
	training_labels, testing_labels = [], []
	test_index = random.sample(range(50), 10)
	for i in range(50):
		if i in test_index:
			testing_set.append(doc_list[i])
			testing_labels.append(labels[i])
		else:
			training_set.append(doc_list[i])
			training_labels.append(labels[i])
	training_matrix = setOfWords2Vec(vocal_list, training_set)
	testing_matrix = setOfWords2Vec(vocal_list, testing_set)
	feature_prob, label_prob = trainNB0(training_matrix, training_labels)
	error = 0
	for i in range(10):
		classified = classifyNB(testing_matrix[i], feature_prob, label_prob)
		if classified != testing_labels[i]:
			error += 1
	return error / 10








