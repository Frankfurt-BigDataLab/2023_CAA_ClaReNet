import os
import cv2
import matplotlib.pyplot as plt
import pandas as pd

def detect_keypoints_and_match(img1, img2, method="ORB", max_distance=45):
	"""
	The implementation follows the guide by OpenCV
	https://docs.opencv.org/3.4/d1/d89/tutorial_py_orb.html
	https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html
	"""
	img1 = cv2.imread(img1,0)          # queryImage
	img2 = cv2.imread(img2,0)

	if method   == 'BRISK':
	    finder = cv2.BRISK_create()
	    kp1, des1 = finder.detectAndCompute(img1,None)
	    kp2, des2 = finder.detectAndCompute(img2,None)

	elif method == 'BRIEF':
	    star = cv2.xfeatures2d.StarDetector_create()
	    # Initiate BRIEF extractor
	    brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
	    kp1 = star.detect(img1,None)
	    # compute the descriptors with BRIEF
	    kp1, des1 = brief.compute(img1, kp1)
	    kp2 = star.detect(img2,None)
	    # compute the descriptors with BRIEF
	    kp2, des2 = brief.compute(img2, kp2)
	    # BFMatcher with default params
	    
	if method   == 'ORB':
	    finder = cv2.ORB_create()
	    kp1, des1 = finder.detectAndCompute(img1,None)
	    kp2, des2 = finder.detectAndCompute(img2,None)
	    
	bf = cv2.BFMatcher(cv2.NORM_HAMMING2)
	matches = bf.match(des1,des2)
	matches = sorted(matches, key = lambda x:x.distance)
	
	good_matches = []
	for m in matches:
	    if m.distance <=max_distance:
	        good_matches.append(m)
	return good_matches, img1, kp1, img2, kp2

def visualise_matches(img1, kp1, img2, kp2, matches):
	img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches,None, flags=2)
	plt.figure(figsize=(15,10))
	plt.imshow(img3)
	plt.show()


def extract_matches_in_directory(data_dir, count=True):
	"""
	Applys detect_keypoints and matches to all elements in a directory. Please make sure that there are only images in the directory,
	else also some exceptions can be added.

	count defines if the sum of matches should be returned or the list of found matches.
	"""
	matches = {}

	#create image dictionary
	for root1, dirs1, files1 in os.walk(data_dir, topdown=False):
	    for name1 in files1:
	                matches[name1] = {}

	# extract matches 
	for root1, dirs1, files1 in os.walk(data_dir, topdown=False):
		for name1 in files1:
			img1 = os.path.join(root1, name1)

			for root2, dirs2, files2 in os.walk(data_dir, topdown=False):
				for name2 in files2:
					img2 = os.path.join(root2, name2)

					if name1 == name2:
						continue

					else:
						if name1 in matches:
							if name2 in matches[name1]:
								continue
							else:

								matches_found = detect_keypoints_and_match(img1, img2)
								if count == True:
									matches[name1][name2] = len(matches_found[0])
									matches[name2][name1] = len(matches_found[0])
								else:
									matches[name1][name2] = matches_found[0]
									matches[name2][name1] = matches_found[0]

	df = pd.DataFrame.from_dict(matches, orient="index")
	df.fillna(0, inplace=True)
	return df