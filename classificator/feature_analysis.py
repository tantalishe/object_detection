import numpy as np
import cv2
import features as ft
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

NUMBER_TRAINING_EXAMPLES = 280
NUMBER_TEST_EXAMPLES = 120
NUMBER_CLASSES = 4
FEATURE_TYPE = "humoments"
VISUALISATION = 0 # 1 - console mean and std for every class and feature; 2 - graphics
file_path_list = ["data/dataset3/scew/", "data/dataset3/nut/", "data/dataset3/profile_40/", "data/dataset3/profile_20/"]
if FEATURE_TYPE == "humoments":
	NUMBER_FEATURES = 7
elif FEATURE_TYPE == "standart":
	NUMBER_FEATURES = 3

data_features = np.zeros((NUMBER_CLASSES, NUMBER_FEATURES, NUMBER_TRAINING_EXAMPLES))
data_analysed = np.zeros((NUMBER_CLASSES, NUMBER_FEATURES, 2))

for target_number in range(NUMBER_CLASSES):

	file_path = file_path_list[target_number]

	for n in range(NUMBER_TRAINING_EXAMPLES):
		
		number = n + 1 # LOADING CONTOUR
		filename = file_path + str(number) + ".npz"
		data = np.load(filename)
		cnt = data['arr_0']

		feature_vector = ft.finding_features(cnt, ftype = FEATURE_TYPE)

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
			print ("for feature number", f, "   M =", "{:.3E}".format(data_analysed[c, f, 0]), "   STD =", "{:.3E}".format(data_analysed[c, f, 1]))
		print()

color = ('xkcd:orangered','xkcd:turquoise','xkcd:lime green','xkcd:indigo')
file_path_list = ["Болт", "Гайка", "Профиль 40 мм", "Профиль 20 мм"]
# f = 1
# print(data_features[1, 1])
for f in range(NUMBER_FEATURES):
	# fig = Figure()
	for c in range(NUMBER_CLASSES):
		plt.figure(1)
		histr, bins, _ = plt.hist(data_features[c, f], facecolor=color[c])
		print(np.sum(np.array(histr/NUMBER_TRAINING_EXAMPLES)))
		print(bins)
		print()
		plt.close()
		plt.figure(2)
		plt.subplot(420+f+1)
		bins_axis = np.zeros((len(bins) + 1))
		bins_axis[0] = bins[0]
		bins_axis[-1] = bins[-1]
		histr = np.insert(histr, 0, 0)
		histr = np.append(histr, 0)
		for i in range(len(bins) - 1):
			bins_axis[i+1] = (bins[i] + bins[i+1]) / 2 
		plt.plot(bins_axis, histr/NUMBER_TRAINING_EXAMPLES, color[c], label=file_path_list[c])
		plt.title('Момент номер '+str(f+1))
		plt.legend()
plt.show()