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

def drawPoint(image, x, y, r, g, b):
	cv2.circle(image,(int(x), int(y)), 3, (int(b), int(g), int(r) ), -1)

image_path = "data/test/ellipse.jpg"
image = cv2.imread(image_path, 1)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
thresh = cv2.bitwise_not(thresh)
_, contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnt = contours[0]
rect = cv2.minAreaRect(cnt)
points = cv2.boxPoints(rect)

x1, y1 = rect[0]
x3, y3 = rect[1]
x2, y2 = centroid(cnt)

for p in points:
	x, y = p
	drawPoint(image, x, y, 255,0,0)


drawPoint(image, x1, y1, 255,0,0) # red center
drawPoint(image, x2, y2, 0,255,0) # green centroid

# cv2.imshow("test", image)
# print(rect[0], rect[1], rect[2])
# print(cv2.boxPoints(rect))
dist = rasst(x1, y1, x2, y2)
dist_correct = dist / max(rect[1])

print(image_path)
print(dist_correct)

cv2.waitKey(0)
cv2.destroyAllWindows()