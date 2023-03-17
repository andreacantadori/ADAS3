import cv2
import numpy as np
from time import time
import camera
import sys
import utilities
from numpy import load
from numpy import save
from numpy import asarray

#--------------------------------------------------------------------------------------------------
def showVertexes(img, show):
#--------------------------------------------------------------------------------------------------
    if show:
        i = 0
        for p in detectedTarget:
            cv2.circle(img,(p[0],p[1]),50,255,-1)
            cv2.putText(img, "{:.0f}".format(i), (p[0],p[1]), cv2.FONT_HERSHEY_SIMPLEX, 4, 255, 4, cv2.LINE_4)
            i = i+1
    return img
#--------------------------------------------------------------------------------------------------
def showCenter(img, show):
#--------------------------------------------------------------------------------------------------
    if show:
        for p in projectedTarget:
            cv2.circle(img,(int(p[0][0]),int(p[0][1])),30,0,-1)
    return img
#--------------------------------------------------------------------------------------------------
def showCoordinates(img, x, y, z, show):
#--------------------------------------------------------------------------------------------------
    if show:
        cv2.putText(img, "{:.0f} {:.0f} {:.0f}".format(x,y,z), (100,100), cv2.FONT_HERSHEY_SIMPLEX, 4, 255, 4, cv2.LINE_4)
    return img
#--------------------------------------------------------------------------------------------------
def showCross(img, show):
#--------------------------------------------------------------------------------------------------
    if show:
        cv2.line(img,(0,1518),(4024,1518),255,3)
        cv2.line(img,(2012,0),(2012,3036),255,3)
    return img

# Graphic options
graphicsShowVertexes = False

# Load camera intrinsic matrix
M_K = load('./Support files/Camera/CameraMtx.npy')
# Load camera distortion matrix
M_DIST = load('./Support files/Camera/CameraDist.npy')
print('M_DIST\n',M_DIST)
# ========>>>> TO DO: Manage the cases of missing matrixes!

# The real (i.e. in world coordinates) is a square of 174x174 mm
# ========>>>> ADJUST THESE SIZES AS APPROPRIATE!
# Note that the z-coordinate in the real target's coordinate system is 0
# See initPlateVertexes for more details
realTarget = utilities.initPlateVertexes(174,174,0)
# The center of the real target has world coordinates (0,0,0)
centerTarget = np.zeros((1,3), np.float32)

# The detected target, as displayed on the 2D screen, is an array of 4 (x,y) points
detectedTarget = np.zeros((4,2), dtype=np.float32)

# Create camera object (encapsulates pypylon) and set the camera to its maximum resolution
cam = camera.Camera(4024,3036)

# Start grabbing
# Note that the camera.latestFrame method is asynchronous, i.e. it always
# returns the latest available frame, so that image grabbing continues 
# even during image transfer via USB
cam.start()
cv2.waitKey(500)
if cam.latestFrame is None:
    sys.exit(1)

# Lists of all circles, squares and target candidates detected in the image
circles = []
squares = []
targets = []

runLoop = True
while runLoop:

    # Get the latest image (the camera is set in 8-bit gray mode for speed of both
    # data transfer via USB and processing time)
    imageGray = cam.latestFrame
    if imageGray is None:
        sys.exit(1)

    # The image is binary thresholded
    # ========>>>> TO DO: introduce global adaptive threshold to account for varying lighting conditions
    _, imageBW = cv2.threshold(imageGray, 80, 255, 0)

    # Find all contours in the image
    contours, _ = cv2.findContours(imageBW, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # Some contours may be square, others may be circles: get all of them into
    # their respective lists
    circles.clear()
    squares.clear()
    for contour in contours:
        # Approximate the contour with a polyline and get its area (in pixels)
        approx = cv2.approxPolyDP(contour,0.01*cv2.arcLength(contour,True),True)
        area = cv2.contourArea(contour)
        # A contour is a square if it approximated by a polyline with 4 segments
        # Very small contours are discarded
        if len(approx) == 4 and area > 1000:
            # OK, this contour is a rectangle. 
            # Let's get the best bounding rect and check if it is really a square
            # by computing the ratio between its width and height (which is ideally 1, 
            # but we allow for some tolerance)
            x, y, w, h = cv2.boundingRect(contour)
            ratio = float(w)/h
            if ratio >= 0.8 and ratio <= 1.2:
                # This contour is really a square: let's add it to the list with all of irs attributes
                squares.append([approx, x, y, w, h, area])
        # We looked for squares. Now we look for circles
        # A contour is considered a circle when:
        #       - its approximating polyline has more than 8 sides
        #       - it is convex
        # As usual, we discard too small contours
        elif len(approx) > 8 and cv2.isContourConvex(approx) and area > 1000:
            # Don't care much about the lines below: it is OpenCV's method to find the center of mass of
            # the contour, which we assume as the center of the circle
            M = cv2.moments(contour)
            x = int(round(M["m10"]/M["m00"]))
            y = int(round(M["m01"]/M["m00"]))
            # Append the circle and its properties to the list
            circles.append([contour, x, y, area])
    winner = None
    for s in squares:
        targets.clear()
        # If a circle lays inside a square, and its area is 2.87 (+/- a tolerance) times 
        # smaller than that of the square, then that square is a candidate to be our target
        for c in circles:
            if c[1] > s[1] and c[1] < s[1]+s[3] and c[2] > s[2] and c[2] < s[2]+s[4] and s[5] < c[3]*2.87*1.2 and s[5] > c[3]*2.87*0.8:
                targets.append(c)
        # If the square contains exactly 1 circle, it is candidate winner
        if len(targets) == 1:
            if winner is None:
                winner = s
            else:
                # If a winner was already nominated, and the new candidate has smaller area than it,
                # then the new candidate becomes the winner
                # In other words: we choose the smallest square which contains one and only one circle
                # whose area is 2.87 times smaller than that of the square
                if s[4] < winner[4]:
                    winner = s
    # Finally we have a winner: our target in the image!
    if winner is not None:
        # The winner is a square, defined by its 4 vertexes. For the projection algorithm to work properly,
        # we need that the vertexes of the square are ordered always the same way, e.g. the first being the 
        # top left, the second the top right, and so on moving clockwise. 
        # This is what is done by the orderPoints() method
        i = 0
        for p in winner[0]:
            detectedTarget[i][0] = p[0][0]
            detectedTarget[i][1] = p[0][1]
            i = i+1
        detectedTarget = utilities.orderPoints(detectedTarget)
        # Call showVertexes(_,True) to highlight the vertexes in the image
        imageGray = showVertexes(imageGray, False) 
        # Solve the projection problem by looking for the rotation
        # and translation vectors from word (3D) to screen (2D) coordinate
        _, rVec, tVec = cv2.solvePnP(realTarget, np.float32(detectedTarget), M_K, M_DIST, cv2.SOLVEPNP_P3P)
        # rVec is the (3x3) rotation matrix 
        # tVec is the (3x1) translation vector 
        projectedTarget, _ =  cv2.projectPoints(centerTarget, rVec, tVec, M_K, M_DIST) 
        showCenter(imageGray, False)
        x = tVec[0][0]
        y = tVec[1][0]
        z = tVec[2][0]
        showCoordinates(imageGray, x, y, z, True)
    showCross(imageGray, False)
    cv2.imshow('Window',imageGray)
    k = cv2.waitKey(10)
    if k == utilities.Key.ESC:
        runLoop = False

cam.close()
cv2.destroyAllWindows()




