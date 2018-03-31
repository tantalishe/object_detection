import numpy as np
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
import pickle

clf = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=5)
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
