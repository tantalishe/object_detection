import cv2
import numpy as np

def get_contour_intensity(original_image, contour):
	mask = np.zeros(original_image.shape[:2], dtype="uint8")
	cv2.drawContours(mask,[contour], -1, 255, -1)
	mean = cv2.mean(original_image, mask)
	return mean

def get_contour_centroid(contour):
	M = cv2.moments(contour)
	cx = int(M['m10']/M['m00'])
	cy = int(M['m01']/M['m00'])
	return cx, cy