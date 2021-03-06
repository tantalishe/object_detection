import cv2
import numpy as np
import math

def getSampleContour(img_path):
	img = cv2.imread(img_path, 1)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	ret, thresh = cv2.threshold(gray,50,255,cv2.THRESH_BINARY_INV)
	# thresh = cv2.bitwise_not(thresh)
	_, contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	contour = contours[0]
	return contour

def centroid(contour):
	M = cv2.moments(contour)
	cx = int(M['m10']/M['m00'])
	cy = int(M['m01']/M['m00'])
	return cx, cy

def dst(x1, y1, x2, y2):
	d = math.hypot(x2 - x1, y2 - y1)
	return d

def finding_hu_M(contour):
	M = cv2.moments(contour)
	HuM = cv2.HuMoments(M)
	return HuM

def f_humoments(cnt):
	feature_vector = []
	HuM = finding_hu_M(cnt)
	for i in range(7):
		feature_vector.append(HuM[i,0])
	return feature_vector

def f_standart(cnt):
	circle = getSampleContour("data/dataset1/test/circle.jpg")

	rect = cv2.minAreaRect(cnt)
	points = cv2.boxPoints(rect)
	x1, y1 = rect[0]
	w, h = rect[1]
	x2, y2 = centroid(cnt)
	dist = dst(x1, y1, x2, y2)

	dist_feature = dist / max(rect[1]) # DISTANCE BETWEEN CENTROID AND CENTER MINAREARECT
	proportion_feature = max(w,h) / min(w,h) # ASPECT RATIO OF MINAREARECT
	eps_circle_feature = cv2.matchShapes(cnt, circle, 1, 0) # SLOZHNO

	feature_vector = [dist_feature, proportion_feature, eps_circle_feature]
	return feature_vector

def finding_features(cnt, ftype = "standart"):

	if ftype == "humoments":
		feature_vector = f_humoments(cnt)

	elif ftype == "standart":
		feature_vector = f_standart(cnt)

	return feature_vector


