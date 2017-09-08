import cv2
import numpy as np
import math

NUMBER_EXAMPLES = 150
NUMBER_CLASSES = 2
file_path_list = ["data/scew/", "data/nut/"]
file_saving_path = "data/"

def centroid(contour):
	M = cv2.moments(contour)
	cx = int(M['m10']/M['m00'])
	cy = int(M['m01']/M['m00'])
	return cx, cy

def rasst(x1, y1, x2, y2):
	d = math.hypot(x2 - x1, y2 - y1)
	return d

def finding_features(cnt):
	rect = cv2.minAreaRect(cnt)
	points = cv2.boxPoints(rect)
	x1, y1 = rect[0]
	w, h = rect[1]
	x2, y2 = centroid(cnt)
	dist = rasst(x1, y1, x2, y2)
	dist_feature = dist / max(rect[1])
	proportion_feature = max(w,h) / min(w,h)
	return dist_feature, proportion_feature


data_list = []
target_list = []

for target_number in range(NUMBER_CLASSES):

	file_path = file_path_list[target_number]

	for n in range(NUMBER_EXAMPLES):
		
		number = n + 1 # LOADING CONTOUR
		filename = file_path + str(number) + ".npz"
		data = np.load(filename)
		cnt = data['arr_0']

		f1, f2 = finding_features(cnt) # FINDING FEATURES

		feature_vector = [f1,f2] # APPEND IT INTO DATASET
		data_list.append(feature_vector)
		target_list.append(target_number)


data_list = np.asarray(data_list) # TRANSFORMING INTO NP FORMAT
target_list = np.asarray(target_list)

# print(data_list)
# print(target_list)
# print(data_list.shape)
# print(target_list.shape)

np.save(file_saving_path + 'cnt_data', data_list) # SAVIN
np.save(file_saving_path + 'cnt_targets', target_list)
