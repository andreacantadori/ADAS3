from __future__ import print_function
import cv2 as cv
import numpy as np
import argparse

MAX_FEATURES = 1000

parser = argparse.ArgumentParser(description='Code for Feature Matching with FLANN tutorial.')
parser.add_argument('--input1', help='Path to input image 1.', default='box.png')
parser.add_argument('--input2', help='Path to input image 2.', default='box_in_scene.png')
args = parser.parse_args()
img_object = cv.imread(cv.samples.findFile(args.input1), cv.IMREAD_GRAYSCALE)
img_scene = cv.imread(cv.samples.findFile(args.input2), cv.IMREAD_GRAYSCALE)
if img_object is None or img_scene is None:
    print('Could not open or find the images!')
    exit(0)

# Detect ORB features and compute descriptors.
orb = cv.ORB_create(MAX_FEATURES)
keypoints_obj, descriptors_obj = orb.detectAndCompute(img_object, None)
keypoints_scene, descriptors_scene = orb.detectAndCompute(img_scene, None)

# create BFMatcher object
bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
# Match descriptors.
matches = bf.match(descriptors_obj,descriptors_scene)
# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)

#-- Draw matches
img_matches = np.empty((max(img_object.shape[0], img_scene.shape[0]), img_object.shape[1]+img_scene.shape[1], 3), dtype=np.uint8)
cv.drawMatches(img_object, keypoints_obj, img_scene, keypoints_scene, matches, img_matches, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
#-- Localize the object
obj = np.empty((len(matches),2), dtype=np.float32)
scene = np.empty((len(matches),2), dtype=np.float32)
for i in range(len(matches)):
    #-- Get the keypoints from the good matches
    obj[i,0] = keypoints_obj[matches[i].queryIdx].pt[0]
    obj[i,1] = keypoints_obj[matches[i].queryIdx].pt[1]
    scene[i,0] = keypoints_scene[matches[i].trainIdx].pt[0]
    scene[i,1] = keypoints_scene[matches[i].trainIdx].pt[1]
H, _ =  cv.findHomography(obj, scene, cv.RANSAC, 5)
#-- Get the corners from the image_1 ( the object to be "detected" )
obj_corners = np.empty((4,1,2), dtype=np.float32)
obj_corners[0,0,0] = 0
obj_corners[0,0,1] = 0
obj_corners[1,0,0] = img_object.shape[1]-1
obj_corners[1,0,1] = 0
obj_corners[2,0,0] = img_object.shape[1]-1
obj_corners[2,0,1] = img_object.shape[0]-1
obj_corners[3,0,0] = 0
obj_corners[3,0,1] = img_object.shape[0]-1
scene_corners = cv.perspectiveTransform(obj_corners, H)
#-- Draw lines between the corners (the mapped object in the scene - image_2 )
cv.line(img_matches, (int(scene_corners[0,0,0] + img_object.shape[1]), int(scene_corners[0,0,1])),\
    (int(scene_corners[1,0,0] + img_object.shape[1]), int(scene_corners[1,0,1])), (0,255,0), 4)
cv.line(img_matches, (int(scene_corners[1,0,0] + img_object.shape[1]), int(scene_corners[1,0,1])),\
    (int(scene_corners[2,0,0] + img_object.shape[1]), int(scene_corners[2,0,1])), (0,255,0), 4)
cv.line(img_matches, (int(scene_corners[2,0,0] + img_object.shape[1]), int(scene_corners[2,0,1])),\
    (int(scene_corners[3,0,0] + img_object.shape[1]), int(scene_corners[3,0,1])), (0,255,0), 4)
cv.line(img_matches, (int(scene_corners[3,0,0] + img_object.shape[1]), int(scene_corners[3,0,1])),\
    (int(scene_corners[0,0,0] + img_object.shape[1]), int(scene_corners[0,0,1])), (0,255,0), 4)
#-- Show detected matches
cv.imshow('Good Matches & Object detection', img_matches)
cv.waitKey()