import os
import numpy as np
import cv2


def apply_circle_crop(src, img_size=224, percentage=0.95):
	image = cv2.imread(src)
	image = cv2.resize(image, (224, 224)) 
	circle_img = np.zeros((img_size, img_size), np.uint8)
	cv2.circle(circle_img, ((int)(img_size/2),(int)(img_size/2)), int(img_size/2*percentage), 1, thickness=-1)
	image = cv2.bitwise_and(image, image, mask=circle_img)

	return image

def circle_crop_directory(src, target, img_size=224, percentage=0.95):
	
	for root, dirs, files in os.walk(src, topdown=False):
	    for name in files:
	        img = os.path.join(root, name)
	        image = apply_circle_crop(img)
	        cv2.imwrite(os.path.join(target, name), image)


def apply_grayscale(src, img_size=512):
	image = cv2.imread(src)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	image = cv2.resize(image, (img_size, img_size), interpolation = cv2.INTER_LINEAR)

	return image

def grayscale_directory(src, target, img_size=224):
	
	for root, dirs, files in os.walk(src, topdown=False):
	    for name in files:
	        img = os.path.join(root, name)
	        image = apply_grayscale(img)
	        cv2.imwrite(os.path.join(target, name), image)




def apply_clahe(src):

    image = cv2.imread(src)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(2, 2))
    equalized = clahe.apply(gray)

    return equalized

def clahe_directory(src, target, img_size=224):
	
	for root, dirs, files in os.walk(src, topdown=False):
	    for name in files:
	        img = os.path.join(root, name)
	        image = apply_clahe(img)
	        cv2.imwrite(os.path.join(target, name), image)