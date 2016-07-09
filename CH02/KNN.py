import numpy as np
import operator
from collections import Counter
from scipy.spatial import distance

def toyDatasets():
	group = np.array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
	labels = ['A','A','B','B']
	return group, labels

def classify0(example, data_set, labels, k):
	# if type(k) != int or k <= 0 or k > len(labels):
	# 	print 'bad k'
	# 	return
	dist_mat = np.array([distance.euclidean(example, row) for row in data_set])
	key_indexes = dist_mat.argsort()[:k]
	label_count = Counter(np.array(labels)[key_indexes])
	sorted_labels = sorted(label_count.items() ,key=operator.itemgetter(1), reverse=True)
	return sorted_labels[0][0]

