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


data_list = [] # MAKE TRAIN DATASET
target_list = []

for target_number in range(NUMBER_CLASSES):

	file_path = file_path_list[target_number]

	for n in range(NUMBER_TRAINING_EXAMPLES):
		
		number = n + 1 # LOADING CONTOUR
		filename = file_path + str(number) + ".npz"
		data = np.load(filename)
		cnt = data['arr_0']

		feature_vector = ft.finding_features(cnt, ftype = FEATURE_TYPE) # FINDING FEATURES AND APPEND IT TO SET
		data_list.append(feature_vector)
		target_list.append(target_number)

data_list = np.asarray(data_list) # TRANSFORMING INTO NP FORMAT
target_list = np.asarray(target_list)
np.save(file_saving_path + 'cnt_data', data_list) # SAVIN
np.save(file_saving_path + 'cnt_targets', target_list)
print("Train dataset done")



test_data_list = [] # MAKE TEST DATASET
test_target_list = []

for target_number in range(NUMBER_CLASSES):

	file_path = file_path_list[target_number]

	for n in range(NUMBER_TEST_EXAMPLES):
		
		number = NUMBER_TRAINING_EXAMPLES + n + 1 # LOADING CONTOUR
		filename = file_path + str(number) + ".npz"
		data = np.load(filename)
		cnt = data['arr_0']

		feature_vector = ft.finding_features(cnt, ftype = FEATURE_TYPE) # FINDING FEATURES AND APPEND IT TO SET
		test_data_list.append(feature_vector)
		test_target_list.append(target_number)

test_data_list = np.asarray(test_data_list) 
test_target_list = np.asarray(test_target_list)
np.save(file_saving_path + 'cnt_test_data', test_data_list) 
np.save(file_saving_path + 'cnt_test_targets', test_target_list)
print("Test dataset done")
