import numpy as np
import pandas as pd
import operator
from collections import Counter
from scipy.spatial import distance


def toyDatasets():
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classify0(example, data_set, labels, k):
    # if type(k) != int or k <= 0 or k > len(labels):
    # 	print 'bad k'
    # 	return
    dist_mat = np.array([distance.euclidean(example, row) for row in data_set])
    key_indexes = dist_mat.argsort()[:k]
    label_count = Counter(np.array(labels)[key_indexes])
    sorted_labels = sorted(label_count.items(), key=operator.itemgetter(1),
                           reverse=True)
    return sorted_labels[0][0]


def file2matrix(filename):
    love_dictionary = {'largeDoses': 3, 'smallDoses': 2, 'didntLike': 1}
    data = pd.read_table(filename, header=None)
    feature_matrix = data.iloc[:, 0:-1].values
    labels = data.iloc[:, -1].map(love_dictionary).values
    return feature_matrix, labels


def autoNorm(dataSet):
    if len(dataSet) <= 1:
        return dataSet
    dataSet = np.array(dataSet)
    min_values = dataSet.min(axis=0).reshape(1, len(dataSet[0]))
    max_values = dataSet.max(axis=0).reshape(1, len(dataSet[0]))
    ranges = max_values - min_values
    normalized_matrix = (dataSet - min_values) / ranges
    return normalized_matrix, ranges, min_values


def datingClassTest(filename, ratio, k):
    data_matrix, labels = file2matrix(filename)
    normalized_matrix, ranges, min_values = autoNorm(data_matrix)
    test_size = int(len(labels) * ratio)
    test_matrix, train_matrix = np.split(normalized_matrix, [test_size])
    test_labels, train_labels = np.split(labels, [test_size])
    error_count = 0
    for i in range(test_size):
        example = test_matrix[i]
        label = test_labels[i]
        classifier_res = classify0(example, train_matrix, train_labels, k)
        if label != classifier_res:
            error_count += 1
    return float(error_count) / test_size


