import numpy as np
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import math
import pickle

data = np.load("data/cnt_data.npy")
target = np.load("data/cnt_targets.npy")

test_data = np.load("data/cnt_test_data.npy")
test_target = np.load("data/cnt_test_targets.npy")

classifiers = [
    KNeighborsClassifier(5),
    SVC(gamma=2, C=1),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=7),
    GaussianNB(),]

names = [ "Nearest neighbors", "SVM", "Random forest", "Naive bayes"]


# print(test_target)
# print(prediction)

for name, clf in zip(names, classifiers):
	clf.fit(data, target)

	prediction = clf.predict(test_data)
	
	n = prediction.size # CHECK CORRECTNESS
	a = 0
	for i in range(n):
		if test_target[i] == prediction[i]:
			a += 1

	correctness = a / n * 100

	print("Correctness for", name, " classifier on test data are ", correctness, "%")