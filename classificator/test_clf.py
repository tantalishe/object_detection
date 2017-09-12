import numpy as np
from sklearn import datasets
from sklearn.svm import SVC
import features as ft
import cv2
import math
import pickle

file = open('saved_model', 'rb')
clf = pickle.load(file)


predict_data = []  # FOR TESTS

data = np.load("data/dataset1/profile_40/158.npz")
cnt = data['arr_0']
f1, f2 = ft.finding_features(cnt)
feature_vector = [f1,f2]
predict_data.append(feature_vector)

data = np.load("data/dataset1/profile_20/156.npz")
cnt = data['arr_0']
f1, f2 = ft.finding_features(cnt)
feature_vector = [f1,f2]
predict_data.append(feature_vector)

data = np.load("data/dataset1/scew/153.npz")
cnt = data['arr_0']
f1, f2 = ft.finding_features(cnt)
feature_vector = [f1,f2]
predict_data.append(feature_vector)


a = clf.predict(predict_data)
print(a)