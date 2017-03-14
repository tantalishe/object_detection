import cv2
import numpy as np
import hist_processing as hp

cap = cv2.VideoCapture(0)
while True:
    _, image = cap.read()
    image = cv2.medianBlur(image, 5)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([image],[0],None,[256],[0,256])
    histogram = hp.get_hist_image(hist)
    cv2.imshow("origin", image)
    cv2.imshow("hist", histogram)

    if cv2.waitKey(1) == 27:
        break
cv2.destroyAllWindows()
cap.release()
