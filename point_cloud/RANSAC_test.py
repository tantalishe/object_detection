import ransac_module as rm
import numpy as np


n = 100
max_iterations = 100
goal_inliers = n * 0.3
threshold = 0.01

xyzs = np.random.random((n, 3)) * 10
xyzs[:50, 2:] = xyzs[:50, :1]

m, b = rm.run_ransac(xyzs, threshold, 3, goal_inliers, max_iterations)
a, b, c, d = m
print(m)