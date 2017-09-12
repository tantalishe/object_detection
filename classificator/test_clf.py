import numpy as np
from sklearn import datasets
from sklearn.svm import SVC
import features as ft
import cv2
import math
import pickle

file = open('saved_model', 'rb')
clf = pickle.load(file)

test_data = np.load("data/cnt_test_data.npy")
test_target = np.load("data/cnt_test_targets.npy")

prediction = clf.predict(test_data)

# print(test_target)
# print(prediction)

n = prediction.size
a = 0
for i in range(n):
	if test_target[i] == prediction[i]:
		a += 1

correctness = a / n * 100

print("Correctness on test data are ", correctness, "%")