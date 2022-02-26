#!/usr/bin/evn python

"""
CMSC733 Spring 2019: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project1: MyAutoPano: Phase 1 Starter Code

Author(s): 
Chahat Deep Singh (chahat@terpmail.umd.edu) 
PhD Student in Computer Science,
University of Maryland, College Park

Nitin J. Sanket (nitinsan@terpmail.umd.edu)
PhD Candidate in Computer Science,
University of Maryland, College Park
"""

# Code starts here:

import numpy as np
import glob
import cv2
import math
from skimage.feature import peak_local_max
# Add any python libraries here

def anms(img, features = 100):
	coords = []
	max_coords = peak_local_max(img, min_distance = 10, threshold_abs = img.mean(), exclude_border = False)
	N_strong = max_coords.shape[0]
	r = float('inf')*np.ones((N_strong, 1))

	for i in range(N_strong):
		for j in range(N_strong):
			if img[max_coords[j, 0], max_coords[j, 1]] > img[max_coords[i, 0], max_coords[i, 1]]:
				dist = (max_coords[j, 1] - max_coords[i, 1])**2 + (max_coords[j, 0] - max_coords[j, 1])**2
				if dist < r[i]:
					r[i] = dist

	idx = np.argsort(r, axis = 0)[::-1]

	max_coords = max_coords[idx]
	max_coords = np.swapaxes(max_coords, 1, 2)
	max_coords = max_coords[:features, :, :]
	
	return max_coords

def feature_descriptor(image, max_coords):
	fv = []
	coords = []
	patch_size = 41
	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	for i in range(max_coords.shape[0]):
		patch_x = (max_coords[i, 1, 0] - patch_size//2).astype(int)
		patch_y = (max_coords[i, 0, 0] - patch_size//2).astype(int)
		patch = image[patch_y:patch_y+patch_size, patch_x:patch_x+patch_size]
		if patch.shape != (patch_size,patch_size):
			continue
		else:
			patch = image[patch_y:patch_y+patch_size, patch_x:patch_x+patch_size]
			blur = cv2.GaussianBlur(patch, (3,3), cv2.BORDER_DEFAULT)
			feature = cv2.resize(patch, (8,8), interpolation=cv2.INTER_AREA)
			fv_i = np.ravel(feature)
			fv_i = (fv_i - np.mean(fv_i))/np.std(fv_i+10**-8)
			coords.append((max_coords[i, 0, 0], max_coords[i, 1, 0]))
			fv.append(fv_i)
	fv = np.array(fv)
	coords = np.array(coords)
	
	return fv, coords.astype(int)

def matching(coords1, coords2, img1, img2):
	keypoint1 = []
	keypoint2 = []
	
	for i in range(coords1.shape[0]):
		diff = np.sum(abs(img1[i,:] - img2), axis = 1)
		idx = np.argsort(diff, axis=0)
		ratio = diff[idx[0]]/diff[idx[1]]
		if ratio < 0.9:
			# import pdb;pdb.set_trace()
			keypoint1.append((coords1[i, 0], coords1[i, 1]))
			keypoint2.append((coords2[idx[0], 0], coords2[idx[0], 1]))
	return np.array(keypoint1).astype(int), np.array(keypoint2).astype(int)

def ransac(kp1, kp2):
	iterations = 0
	inliers = 0
	kp1[:,[0,1]]= kp1[:,[1,0]]
	kp2[:,[0,1]]= kp2[:,[1,0]]
	ones = np.ones((kp1.shape[0], 1))
	dp2 = np.hstack((kp2,ones))
	dp1 = np.hstack((kp1,ones))
	points = []
	inset = []
	while iterations < 1000:
		inliers = 0
		idx = np.random.choice(kp1.shape[0], 4, replace=True)
		p1 = kp1[idx]
		p2 = kp2[idx]
		points.append(idx)
		H, st1 = cv2.findHomography(p1, p2)
		ssd = []
		if st1.all():
			for i in range(kp1.shape[0]):
				point1 = dp1[i]
				dist = (H.dot(point1.T) - dp2[i])
				dist = float(np.sum(np.square(dist)))
				if dist<500:
					inliers = inliers + 1
		
			inset.append(inliers)
			iterations = iterations + 1
	inset = np.array(inset)
	t = np.argsort(inset)[::-1]
	points = np.array(points)
	best = points[t[0]]

	p1 = kp1[best]
	p2 = kp2[best]
	H_til, st2 = cv2.findHomography(p1, p2)
	new_kp1 = []
	new_kp2 = []
	if st2.all():
		for i in range(kp1.shape[0]):
			point1 = dp1[i]
			dist = (H_til.dot(point1.T) - dp2[i])
			dist = float(np.sum(np.square(dist)))
			# print(dist)
			if dist<300:
				new_kp1.append(kp1[i])
				new_kp2.append(kp2[i])
	# import pdb;pdb.set_trace()
	H_cap, st2 = cv2.findHomography(np.array(new_kp1), np.array(new_kp2))
	new_kp1 = np.array(new_kp1)
	new_kp2 = np.array(new_kp2)
	new_kp1[:,[0,1]]= new_kp1[:,[1,0]]
	new_kp2[:,[0,1]]= new_kp2[:,[1,0]]
	return new_kp1, new_kp2, H_til

def drawMatches(img1, img2, k1, k2):
	new = np.concatenate((img1,img2),axis=1)
	k2[:,1] = k2[:,1] + img2.shape[1]
	for i in range(k1.shape[0]):
		cv2.line(new, (k1[i,1],k1[i,0]), (k2[i,1],k2[i,0]), [0,255,255], 1)
		cv2.circle(new, (k1[i,1],k1[i,0]), 3, [0,0,255])
		cv2.drawMarker(new, (k2[i,1],k2[i,0]), [0,255,0], markerSize=5)
	return new

def overlay(base, warp):
	for i in range(warp.shape[0]):
		for j in range(warp.shape[1]):
			if base[i,j].any() > 0:
				warp[i,j] = base[i,j]
	return warp

def pano_size(imgs, H):
	shape = imgs[0].shape
	corner = np.array([[0,0,1],[shape[1]-1,0,1],[0,shape[0]-1,1],[shape[1], shape[0],1]])
	points = []
	for i in range(len(H)):
		for j in range(corner.shape[0]):
			tran_H = H[i]@corner[j].T
			tran_H = tran_H/tran_H[2]
			# import pdb; pdb.set_trace()
			points.append([tran_H[0], tran_H[1]])
	points = np.array(points)
	x_max, x_min = np.max(points[:,0]), np.min(points[:,0])
	y_max, y_min = np.max(points[:,1]), np.min(points[:,1])

	return abs(y_max-y_min).astype(int), abs(x_max-x_min).astype(int), x_min.astype(int), y_min.astype(int)

def create_pano(imgs, H):
	height, width, x_min, y_min = pano_size(imgs, H)
	iter_img = imgs.copy()
	anchor = iter_img.pop(len(imgs)//2)
	I = np.eye(3)
	I[0,2] += -x_min
	I[1,2] += -y_min
	pano = cv2.warpPerspective(anchor, I, (width, height))
	for i in range(len(H)):
		H[i][0,2] += -x_min
		H[i][1,2] += -y_min
		warped = cv2.warpPerspective(iter_img[i], H[i], (width, height))
		pano = overlay(pano, warped)
	return pano

def get_homographies(H):
	start = len(H)//2
	new_H = H.copy()
	if len(H) > 2:
		for a in range(0, start):
			for b in range(a+1, start+1):
				print(a,b)
				new_H[a] = new_H[a].copy()@new_H[b].copy()
	# import pdb; pdb.set_trace()


	if len(H) > 3:
		for m in range(len(H)-1, start+1, -1):
			for n in range(m-1, start, -1):
				print(m,n)
				new_H[m] = new_H[m].copy()@new_H[n].copy()
	return new_H

def main():
	# Add any Command Line arguments here
    # Parser = argparse.ArgumentParser()
    # Parser.add_argument('--NumFeatures', default=100, help='Number of best features to extract from each image, Default:100')
    
    # Args = Parser.parse_args()
    # NumFeatures = Args.NumFeatures

	"""
	Read a set of images for Panorama stitching
	"""
	imgs = []
	path = '../Data/Train/Set2/' # Change path here
	for i in range(1,4): # Change number of images read here
		imgs.append(cv2.imread(path+str(i)+'.jpg'))

	centre = len(imgs)//2
	"""
	Corner Detection
	Save Corner detection output as corners.png
	"""
	i = centre
	Homographies = []
	while i > 0:
		image1 = imgs[i].copy()
		image2 = imgs[i-1].copy()
		gray1 = cv2.cvtColor(image1.copy(), cv2.COLOR_BGR2GRAY)
		gray2 = cv2.cvtColor(image2.copy(), cv2.COLOR_BGR2GRAY)

		chcd1 = cv2.cornerHarris(gray1.copy(), 2, 3, 0.001)
		chcd2 = cv2.cornerHarris(gray2.copy(), 2, 3, 0.001)

		"""
		Perform ANMS: Adaptive Non-Maximal Suppression
		Save ANMS output as anms.png
		"""
		hcd1 = anms(chcd1.copy(),500).astype(int)
		hcd2 = anms(chcd2.copy(),500).astype(int)

		"""
		Feature Descriptors
		Save Feature Descriptor output as FD.png
		"""

		fv1, new_hcd1 = feature_descriptor(image1.copy(), hcd1.copy())
		fv2, new_hcd2 = feature_descriptor(image2.copy(), hcd2.copy())

		"""
		Feature Matching
		Save Feature Matching output as matching.png
		"""
		k1, k2 = matching(new_hcd1.copy(), new_hcd2.copy(), fv1.copy(), fv2.copy())

		"""
		Refine: RANSAC, Estimate Homography
		"""
		kp1, kp2, H = ransac(k2.copy(), k1.copy())
		Homographies.insert(0,H)
		i -= 1
		
	i = centre
	while i < len(imgs)-1:
		image1 = imgs[i].copy()
		image2 = imgs[i+1].copy()
		gray1 = cv2.cvtColor(image1.copy(), cv2.COLOR_BGR2GRAY)
		gray2 = cv2.cvtColor(image2.copy(), cv2.COLOR_BGR2GRAY)

		chcd1 = cv2.cornerHarris(gray1.copy(), 2, 3, 0.001)
		chcd2 = cv2.cornerHarris(gray2.copy(), 2, 3, 0.001)
		
		"""
		Perform ANMS: Adaptive Non-Maximal Suppression
		Save ANMS output as anms.png
		"""
		hcd1 = anms(chcd1.copy(),500).astype(int)
		hcd2 = anms(chcd2.copy(),500).astype(int)

		"""
		Feature Descriptors
		Save Feature Descriptor output as FD.png
		"""

		fv1, new_hcd1 = feature_descriptor(image1.copy(), hcd1.copy())
		fv2, new_hcd2 = feature_descriptor(image2.copy(), hcd2.copy())

		"""
		Feature Matching
		Save Feature Matching output as matching.png
		"""
		k1, k2 = matching(new_hcd1.copy(), new_hcd2.copy(), fv1.copy(), fv2.copy())

		"""
		Refine: RANSAC, Estimate Homography
		"""
		kp1, kp2, H = ransac(k2.copy(), k1.copy())
		Homographies.append(H)
		i += 1
	"""
	Image Warping + Blending
	Save Panorama output as mypano.png
	"""
	new_H = get_homographies(Homographies)
	pano = create_pano(imgs, new_H)
	cv2.imwrite('mypano.png', pano)
    
if __name__ == '__main__':
    main()
 
