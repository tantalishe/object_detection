# Original code taken from https://github.com/falcondai/py-ransac.git
import numpy as np
import random

def augment(xyzs):
	axyz = np.ones((len(xyzs), 4))
	axyz[:, :3] = xyzs
	return axyz

def estimate(xyzs):
	axyz = augment(xyzs[:3])
	return np.linalg.svd(axyz)[-1][-1, :]

def is_inlier(coeffs, xyz, threshold):
	return np.abs(coeffs.dot(augment([xyz]).T)) < threshold

def run_ransac(data, threshold, sample_size, goal_inliers, max_iterations, stop_at_goal=True, random_seed=None):
	best_ic = 0
	best_model = None
	random.seed(random_seed)
	for i in range(max_iterations):
		s = random.sample(range(data.shape[0]), int(sample_size)) # take random points from point cloud
		s[0] = data[s[0]]
		s[1] = data[s[1]]
		s[2] = data[s[2]]
		m = estimate(s) # calculate plane
		ic = 0
		for j in range(len(data)): # check another points for inliers 
			if is_inlier(m, data[j], threshold): # and calculate number of inliers
				ic += 1

		# print s
		# print 'estimate:', m,
		# print '# inliers:', ic

		if ic > best_ic: # take plane with largest number of inliers
			best_ic = ic
			best_model = m
			if ic > goal_inliers and stop_at_goal:
				break
	# print ('took iterations:', i+1, 'best model:', best_model, 'explains:', best_ic)
	return best_model, best_ic

