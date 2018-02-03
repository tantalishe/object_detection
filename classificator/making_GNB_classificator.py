import numpy as np
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
import pickle

clf = GaussianNB()
data = np.load("data/cnt_data.npy")
# print(data)
# print(data.shape)
target = np.load("data/cnt_targets.npy")
# print(target)
# print(target.shape)

clf.fit(data, target)

file = open('saved_model', 'wb')
pickle.dump(clf, file)
file.close()
