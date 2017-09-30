import numpy as np
import cv2
import features as ft

NUMBER_TRAINING_EXAMPLES = 150
NUMBER_TEST_EXAMPLES = 10
NUMBER_CLASSES = 4
NUMBER_FEATURES = 3  # TODO: making this adaptive
VISUALISATION = 1 # TODO make graphics (2)
file_path_list = ["data/dataset1/scew_test/", "data/dataset1/nut/", "data/dataset1/profile_20/", "data/dataset1/profile_40/"]

data_features = np.zeros((NUMBER_CLASSES, NUMBER_FEATURES, NUMBER_TRAINING_EXAMPLES))
data_analysed = np.zeros((NUMBER_CLASSES, NUMBER_FEATURES, 2))

for target_number in range(NUMBER_CLASSES):

	file_path = file_path_list[target_number]

	for n in range(NUMBER_TRAINING_EXAMPLES):
		
		number = n + 1 # LOADING CONTOUR
		filename = file_path + str(number) + ".npz"
		data = np.load(filename)
		cnt = data['arr_0']

		feature_vector = ft.finding_features(cnt)

		for f in range(NUMBER_FEATURES):
			data_features[target_number, f, n] = feature_vector[f]

for c in range(NUMBER_CLASSES):
	for f in range(NUMBER_FEATURES):
		data_analysed[c, f, 0] = np.mean(data_features[c, f, :]) # index 0 - mean
		data_analysed[c, f, 1] = np.std(data_features[c, f, :]) # index 1 - standart deviation

if VISUALISATION == 1:
	for c in range(NUMBER_CLASSES):
		print()
		print(file_path_list[c])
		for f in range(NUMBER_FEATURES):
			print ("for feature number", f, "   M =", data_analysed[c, f, 0], "   STD =", data_analysed[c, f, 1])
		print()

