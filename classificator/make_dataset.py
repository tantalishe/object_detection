import cv2
import time
import numpy as np

SHOT_TIME = 0.200 # delay between shots [sec]
BLUR = 3
THRESHOLD_KERNEL = 11
THRESHOLD_PARAMETER = 3
DILATE_ITER = 4
ERODE_ITER = 2
DATA_PATH = 'data/dataset1/scew_test/' # path to saving contours

cam = cv2.VideoCapture(1)
number = 0
while True:
    _, frame = cam.read()

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # READING AND FILTERING IMAGE
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

    for cnt in contours: #SAVIN LARGE CONTOURS
        area = cv2.contourArea(cnt)
        # print(area)
        if area > 500:
            cv2.drawContours(frame, cnt, -1, (0, 255, 0),2)
            l = cv2.arcLength(cnt, True)
            cnt = cv2.approxPolyDP(cnt, 0.005 * l, True)
            name = DATA_PATH + str(number)
            number += 1
            np.savez(name, cnt)
            cv2.drawContours(frame, cnt, -1, (0, 0, 255), 3)
            print('shot done  ', number)

    cv2.imshow("frame", frame)

    time.sleep(SHOT_TIME)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cam.release()
cv2.destroyAllWindows()        