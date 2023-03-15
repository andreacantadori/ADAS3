import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['figure.figsize'] = (6.0, 6.0)
matplotlib.rcParams['image.cmap'] = 'gray'

MAX_FEATURES = 1000
MIN_MATCH_COUNT = 10

# Read both images in grayscale mode
img1 = cv2.imread('Ricola.png')
img1Gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2 = cv2.imread('RicolaScene.png')
img2Gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# Detect ORB features and compute descriptors.
orb = cv2.ORB_create(MAX_FEATURES)
keypoints1, descriptors1 = orb.detectAndCompute(img1Gray, None)
keypoints2, descriptors2 = orb.detectAndCompute(img2Gray, None)

FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

matches = flann.knnMatch(np.float32(descriptors1),np.float32(descriptors2),k=2)

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
#     Find points in the Cluttered image corresponding to the target
    dst = cv2.perspectiveTransform(pts,M)
#     Draw a red box around the detected target
    img2 = cv2.polylines(img2,[np.int32(dst)],True,(0,0,255),10, cv2.LINE_AA)
else:
    print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
    matchesMask = None
draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)
img3 = cv2.drawMatches(img1,keypoints1,img2,keypoints2,good,None,**draw_params)

plt.figure(figsize=(12,12))
plt.imshow(img3[...,::-1]),plt.show()
