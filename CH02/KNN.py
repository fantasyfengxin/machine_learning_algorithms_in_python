import numpy as np
import pandas as pd
import operator
from collections import Counter
from scipy.spatial import distance
from os import listdir


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


def img2vector(filename):
    return_matrix = []
    with open(filename, 'r') as file:
        for i in range(32):
            line = file.readline()
            for j in range(32):
                return_matrix.append(int(line[j]))
    return np.array(return_matrix)


def handwritingClassTest():
    error_count = 0
    labels = []
    train_file_list = listdir('trainingDigits')
    train_size = len(train_file_list)
    train_matrix = np.zeros((train_size, 1024))
    for i in range(train_size):
        file_name = train_file_list[i]
        label = int(train_file_list[i].split('_')[0])
        labels.append(label)
        full_file_name = 'trainingDigits/' + file_name
        train_matrix[i,:] = img2vector(full_file_name)

    # process the data for testing
    test_file_list = listdir('testDigits')
    test_size = len(test_file_list)
    for i in range(test_size):
        file_name = test_file_list[i]
        label = int(file_name.split('_')[0])
        full_file_name = 'testDigits/' + file_name
        test_data = img2vector(full_file_name)
        return_label = classify0(test_data, train_matrix, labels, 3)
        if return_label != label:
            error_count += 1
    print 'the error rate is ', (float(error_count) / test_size)