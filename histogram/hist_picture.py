import cv2
import numpy as np
import hist_processing as hp

image_path = "data/test.jpeg"

image = cv2.imread(image_path, 1)

# print image.shape


cv2.imshow("original", image)
blurred = cv2.medianBlur(image, 5)
gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
hist = cv2.calcHist([gray],[0],None,[256],[0,256])
histogram = hp.get_hist_image(hist)
cv2.imshow("processed", gray)
cv2.imshow("hist", histogram)


# print hist.tolist().index(max(hist))

cv2.waitKey(0)
cv2.destroyAllWindows()