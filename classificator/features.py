import cv2
import numpy as np
import math

def centroid(contour):
	M = cv2.moments(contour)
	cx = int(M['m10']/M['m00'])
	cy = int(M['m01']/M['m00'])
	return cx, cy

def rasst(x1, y1, x2, y2):
	d = math.hypot(x2 - x1, y2 - y1)
	return d

def finding_features(cnt):
	rect = cv2.minAreaRect(cnt)
	points = cv2.boxPoints(rect)
	x1, y1 = rect[0]
	w, h = rect[1]
	x2, y2 = centroid(cnt)
	dist = rasst(x1, y1, x2, y2)
	dist_feature = dist / max(rect[1])
	proportion_feature = max(w,h) / min(w,h)
	return dist_feature, proportion_feature