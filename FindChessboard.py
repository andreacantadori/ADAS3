import cv2
import numpy as np
from time import time


# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((4*4,3), np.float32)
objp[:,:2] = np.mgrid[0:4,0:4].T.reshape(-1,2)
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

# Load the image
img = cv2.imread("SceneA0.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Find the chess board corners
time0 = time()
ret, corners = cv2.findChessboardCorners(gray, (3,3), None)
print(time()-time0)
# If found, add object points, image points (after refining them)
if ret == True:
	objpoints.append(objp)
	corners2 = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
	imgpoints.append(corners2)
	# Draw and display the corners
	cv2.drawChessboardCorners(img, (3,3), corners2, ret)
	cv2.imshow('img', img)
	cv2.waitKey(0)
cv2.destroyAllWindows()