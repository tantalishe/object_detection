import cv2
import numpy as np
import contour_processing as cp
import hist_processing as hp

cam = cv2.VideoCapture(0)
# if cam.isOpened() == False:
# 	cam = cv2.VideoCapture(0)
# 	if cam.isOpened() == False:
# 		raise ValueError('Failed to open a capture object.')

invert = 1

while True:
	_, image = cam.read()

	# Blurring

	# image = cv2.GaussianBlur(image, (5,5), 0)
	image = cv2.medianBlur(image, 5)

	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	# Thresholding

	threshold = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,11,2)
	# threshold = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
	# _, threshold = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

	if invert == 1:
		threshold = cv2.bitwise_not(threshold)

	_, contours, hierarchy = cv2.findContours(threshold.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	mask = np.zeros(image.shape[:2], dtype="uint8")

	for contour in contours:
		if cv2.contourArea(contour) < 700:
			continue
		
		mean = cp.get_contour_intensity(gray, contour)
		x, y = cp.get_contour_centroid(contour)
		cv2.drawContours(mask,[contour], -1, 255, -1)
		cv2.drawContours(image,[contour], -1, (0,255,0), 3)
		cv2.putText(image, str(int(mean[0])), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 50, 0), 2)

	hist = cv2.calcHist([gray],[0],None,[256],[0,256])
	histogram = hp.get_hist_image(hist)

	cv2.imshow("original", image)
	# cv2.imshow("grayscale", gray)
	cv2.imshow("mask", mask)
	cv2.imshow("hist", histogram)

	if cv2.waitKey(1) == 27:
		break
cv2.destroyAllWindows()
cam.release()