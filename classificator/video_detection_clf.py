import cv2
import numpy as np
from sklearn import datasets
from sklearn.svm import SVC
import features as ft
import math
import pickle
import time

BLUR = 3
THRESHOLD_KERNEL = 7 # 11
THRESHOLD_PARAMETER = 4 # 3
DILATE_ITER = 4
ERODE_ITER = 2
FEATURE_TYPE = "humoments"
FPS_VISUALISATION = True
FPS_VISUALISATION_DELAY = 3 # [sec]


start_time = time.time()
frame_counter = 0
file = open('saved_model', 'rb')  # loading saved classificator
clf = pickle.load(file)
cam = cv2.VideoCapture(1)
number = 1
while True:
    _, frame = cam.read()

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # reanding and filterinf image
    image = cv2.medianBlur(image, BLUR)
    thres = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,THRESHOLD_KERNEL,THRESHOLD_PARAMETER)
    v = np.median(image)
    sigma = 0.33
    canny_low = int(max(0, (1 - sigma) * v))
    canny_high = int(min(255, (1 + sigma) * v))
    edged = cv2.Canny(thres, canny_low, canny_high)
    edged = cv2.dilate(edged, None, iterations=DILATE_ITER)
    edged = cv2.erode(edged, None, iterations=ERODE_ITER)

    _, contours, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        # print(area)
        if area > 500:

            predict_data = []  # recognizing large contoours
            feature_vector = ft.finding_features(cnt, ftype = FEATURE_TYPE)
            predict_data.append(feature_vector)
            object_id = clf.predict(predict_data)

            if object_id[0] == 0:
                text = 'scew'
            elif object_id[0] == 1:
                text = 'nut'
            elif object_id[0] == 2:
                text = 'profile_20'
            elif object_id[0] == 3:
                text = 'profile_40'

            x, y = ft.centroid(cnt)
            text_pos = (x, y)
            cv2.drawContours(frame, cnt, -1, (255, 255, 0),2)
            cv2.putText(frame, text, text_pos,
                            cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8,
                            color=(0, 255, 0), thickness=2)
    frame_counter += 1    
    if FPS_VISUALISATION and (time.time() - start_time) > FPS_VISUALISATION_DELAY:
        print("FPS:", frame_counter / (time.time() - start_time))
        start_time = time.time()
        frame_counter = 0

    cv2.imshow("frame", frame)
    cv2.imshow("thres", thres)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cam.release()
cv2.destroyAllWindows()        