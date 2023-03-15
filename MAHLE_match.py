import numpy as np
import cv2
import matplotlib.pyplot as plt

MAX_FEATURES = 1000
MIN_MATCH_COUNT = 30
# Read both images in grayscale mode
target = cv2.imread('Target2.png', cv2.IMREAD_GRAYSCALE)
scene = cv2.imread('Scene3.png', cv2.IMREAD_GRAYSCALE)

# Initiate ORB detector
orb = cv2.ORB_create(MAX_FEATURES)

# find the keypoints and compute their descriptors with ORB
kpTarget, desTarget = orb.detectAndCompute(target,None)
kpScene, desScene = orb.detectAndCompute(scene,None)

# draw only keypoints location,not size and orientation
imgTarget = cv2.drawKeypoints(target, kpTarget, None, color=(0,0,255), flags=0)
imgScene = cv2.drawKeypoints(scene, kpScene, None, color=(0,0,255), flags=0)

# create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
# Match descriptors.
matches = bf.match(desTarget,desScene)
# Sort them in the order of their distance
#matches = sorted(matches, key = lambda x:x.distance)[:20]
imgMatches = cv2.drawMatches(imgTarget,kpTarget,imgScene,kpScene,matches,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

ptsTarget = np.float32([kpTarget[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
ptsScene = np.float32([kpScene[m.trainIdx].pt for m in matches]).reshape(-1,1,2)
 
M, _ = cv2.findHomography(ptsTarget, ptsScene, cv2.RANSAC,5.0)
h,w,_ = imgTarget.shape
# Points in the original image
pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
dst = cv2.perspectiveTransform(pts,M)
# Draw a red box around the detected book
imgMatches = cv2.polylines(imgMatches,[np.int32(dst)],True,(0,0,255),10, cv2.LINE_AA)

cv2.imshow('Win1',imgMatches)
cv2.waitKey(0)
quit()


# store all the good matches as per Lowe's ratio test.
good = []
for m,n in matches:
    if m.distance < 0.9*n.distance:
        good.append(m)
if len(good)>MIN_MATCH_COUNT:
    src_pts = np.float32([ keypoints1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ keypoints2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()
    h,w,d = img1.shape
#     Points in the original image
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
#     Find points in the Cluttered image corresponding to the book
    dst = cv2.perspectiveTransform(pts,M)
#     Draw a red box around the detected book
    img2 = cv2.polylines(img2,[np.int32(dst)],True,(0,0,255),10, cv2.LINE_AA)
else:
    print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
    matchesMask = None
draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)
img3 = cv2.drawMatches(img1,keypoints1,img2,keypoints2,good,None,**draw_params)
cv2.namedWindow('Win')
cv2.imshow('Win', img3)
cv2.waitKey(0)
cv2.destroyAllWindows()