import numpy as np
from sklearn import datasets
from sklearn.svm import SVC
import pickle

clf = SVC()
data = np.load("data/cnt_data.npy")
# print(data)
# print(data.shape)
target = np.load("data/cnt_targets.npy")
# print(target)
# print(target.shape)

clf.fit(data, target)

file = open('saved_model', 'wb')
pickle.dump(clf, file)