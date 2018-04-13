import cv2
import numpy as np
import math
import features as ft

NUMBER_TRAINING_EXAMPLES = 150
NUMBER_TEST_EXAMPLES = 10
NUMBER_CLASSES = 4
FEATURE_TYPE = "humoments"
file_path_list = ["data/dataset1/scew_test/", "data/dataset1/nut/", "data/dataset1/profile_20/", "data/dataset1/profile_40/"]
file_saving_path = "data/"


data_list = [] # MAKE DATASET
target_list = []
NUMBER_EXAMPLES = NUMBER_TRAINING_EXAMPLES + NUMBER_TEST_EXAMPLES

for target_number in range(NUMBER_CLASSES):

	file_path = file_path_list[target_number]

	for n in range(NUMBER_EXAMPLES):
		
		number = n + 1 # LOADING CONTOUR
		filename = file_path + str(number) + ".npz"
		data = np.load(filename)
		cnt = data['arr_0']

		feature_vector = ft.finding_features(cnt, ftype = FEATURE_TYPE) # FINDING FEATURES AND APPEND IT TO SET
		data_list.append(feature_vector)
		target_list.append(target_number)

data_list = np.asarray(data_list) # TRANSFORMING INTO NP FORMAT
target_list = np.asarray(target_list)
s = np.arange(NUMBER_EXAMPLES*NUMBER_CLASSES) # SHUFFLE FULL DATASET
data_list[:] = data_list[s[:]]
target_list[:] = target_list[s[:]]

train_data_list = data_list[:NUMBER_TRAINING_EXAMPLES] # SEPARATE FULL SET TO
train_target_list = target_list[:NUMBER_TRAINING_EXAMPLES] # TRAIN AND TEST SETS
test_data_list = data_list[-NUMBER_TEST_EXAMPLES:]
test_target_list = target_list[-NUMBER_TEST_EXAMPLES:]


np.save(file_saving_path + 'cnt_data', data_list) # SAVIN
np.save(file_saving_path + 'cnt_targets', target_list)
print("Train dataset done")

np.save(file_saving_path + 'cnt_test_data', test_data_list) 
np.save(file_saving_path + 'cnt_test_targets', test_target_list)
print("Test dataset done")
