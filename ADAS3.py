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
            cv2.putText(img, "{:.0f}".format(i), (p[0]-25,p[1]+25), cv2.FONT_HERSHEY_SIMPLEX, 2, 0, 4, cv2.LINE_4)
            i = i+1
    return img
#--------------------------------------------------------------------------------------------------
def showCenter(img, pts, show):
#--------------------------------------------------------------------------------------------------
    if show:
        for p in pts:
            cv2.circle(img,(int(p[0][0]),int(p[0][1])),30,0,-1)
    return img
#--------------------------------------------------------------------------------------------------
def showCoordinates(img, coord, pos, show):
#--------------------------------------------------------------------------------------------------
    if show:
        cv2.putText(img, "X,Y,Z (mm): {:.0f},{:.0f},{:.0f}".format(coord[0],coord[1],coord[2]), (pos[0],pos[1]), cv2.FONT_HERSHEY_SIMPLEX, 4, 255, 4, cv2.LINE_4)
    return img
#--------------------------------------------------------------------------------------------------
def showAngles(img, angles, pos, show):
#--------------------------------------------------------------------------------------------------
    if show:
        cv2.putText(img, "Roll,Pitch,Yaw (deg): {:.1f},{:.1f},{:.1f}".format(angles[0],angles[1],angles[2]), (pos[0],pos[1]), cv2.FONT_HERSHEY_SIMPLEX, 4, 255, 4, cv2.LINE_4)
    return img
#--------------------------------------------------------------------------------------------------
def showCross(img, show):
#--------------------------------------------------------------------------------------------------
    if show:
        cv2.line(img,(0,1518),(4024,1518),255,3)
        cv2.line(img,(2012,0),(2012,3036),255,3)
    return img
#--------------------------------------------------------------------------------------------------
def showThreshold(img, thresh, pos, show):
#--------------------------------------------------------------------------------------------------
    if show:
        cv2.putText(img, "Thresh: {:.0f}".format(thresh), (pos[0],pos[1]), cv2.FONT_HERSHEY_SIMPLEX, 4, 255, 4, cv2.LINE_4)
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

# Ratio between the areas of the target square and its inner circle
#kSC = 2.87  # Target M1
kSC = 1.99  # Target M2
kSCTolerance = 0.2  
kSCMin = kSC * (1-kSCTolerance)
kSCMax = kSC * (1+kSCTolerance)

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

# Threshold for binary image thresholding
# We implement a dinamic thresholding algorithm, by adjusting the threshold
# whenever the target cannot be detected. 
lightThreshold = None

#myServer = server.httpServerRequestHandler()
#myServer.run()

runLoop = True
while runLoop:

    # Get the latest image (the camera is set in 8-bit gray mode for speed of both
    # data transfer via USB and processing time)
    imageGray = cam.latestFrame
    if imageGray is None:
        break

    # The image is binary thresholded
    # If no threshold has been established yet, we apply Otsu's optimal threshold detection.    
    if lightThreshold is None:
        lightThreshold, imageBW = cv2.threshold(imageGray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    else:
        _, imageBW = cv2.threshold(imageGray, lightThreshold, 255, cv2.THRESH_BINARY)
    showThreshold(imageGray,lightThreshold,(100,400),True)
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
    targetDetected = None
    for s in squares:
        targets.clear()
        # If a circle lays inside a square, and its area is 2.87 (+/- a tolerance) times 
        # smaller than that of the square, then that square is a candidate to be our target
        for c in circles:
            if c[1] > s[1] and c[1] < s[1]+s[3] and c[2] > s[2] and c[2] < s[2]+s[4] and s[5] < c[3]*kSCMax and s[5] > c[3]*kSCMin:
                targets.append(c)
        # If the square contains exactly 1 circle, it is candidate targetDetected
        if len(targets) == 1:
            if targetDetected is None:
                targetDetected = s
            else:
                # If a targetDetected was already nominated, and the new candidate has smaller area than it,
                # then the new candidate becomes the targetDetected
                # In other words: we choose the smallest square which contains one and only one circle
                # whose area is 2.87 times smaller than that of the square
                if s[4] < targetDetected[4]:
                    targetDetected = s
    if targetDetected is None:
        # We could not find any targetDetected, i.e. any target
        # Therefore we adjust our binary threshold by lowering it by 10% and retry...
        # Once the threshold is too low, we restart from the beginning
        lightThreshold = int(lightThreshold * 0.9)
        if lightThreshold <= 1:
            lightThreshold = None
    else:
        # Finally we have a targetDetected: our target in the image!
        # The targetDetected is a square, defined by its 4 vertexes. For the projection algorithm to work properly,
        # we need that the vertexes of the square are ordered always the same way, e.g. the first being the 
        # top left, the second the top right, and so on moving clockwise. 
        # This is what is done by the orderPoints() method
        i = 0
        for p in targetDetected[0]:
            detectedTarget[i][0] = p[0][0]
            detectedTarget[i][1] = p[0][1]
            i = i+1
        detectedTarget = utilities.orderPoints(detectedTarget)
        # Call showVertexes(_,True) to highlight the vertexes in the image
        imageGray = showVertexes(imageGray, True) 
        # Solve the projection problem by looking for the rotation
        # and translation vectors from word (3D) to screen (2D) coordinate
        _, rVec, tVec = cv2.solvePnP(realTarget, np.float32(detectedTarget), M_K, M_DIST, cv2.SOLVEPNP_P3P)
        # rVec is the (3x1) rotation vector 
        # tVec is the (3x1) translation vector 
        x = tVec[0][0]
        y = tVec[1][0]
        z = tVec[2][0]
        showCoordinates(imageGray, (x,y,z), (100,100), True)
        # Calculate the pitch, yaw, and roll angles
        # Pitch is rotation around x-axis, yaw is rotation around y-axis, and roll is rotation around z-axis
        rot_mat = cv2.Rodrigues(rVec)
        r = rot_mat[0]
        print(r)
        roll = np.arctan2(r[1, 0], r[0, 0])
        pitch = np.arctan2(r[2, 1], r[2, 2])
        yaw = np.arctan2(-r[2, 0], np.sqrt(r[2, 1]**2 + r[2, 2]**2))
        # Convert the angles from radians to degrees
        roll_deg = np.rad2deg(roll)
        pitch_deg = np.rad2deg(pitch)
        if pitch_deg > 0:
            pitch_deg = 180 - pitch_deg
        else:
            pitch_deg = -pitch_deg -180
        yaw_deg = np.rad2deg(yaw)
        showAngles(imageGray, (roll_deg,pitch_deg,yaw_deg), (100,250), True)
    showCross(imageGray, True)
    cv2.imshow('Window',imageGray)
    k = cv2.waitKey(10)
    if k == utilities.Key.ESC:
        runLoop = False
cam.stop()
cv2.waitKey(1000)
cam.close()
cv2.destroyAllWindows()




