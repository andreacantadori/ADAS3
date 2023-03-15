import cv2
import numpy as np
from time import time
import camera
import sys
import Utilities
from numpy import load


# Set to True to highlight the target circles
drawCircles = False
builtinCamera = False

realTarget = Utilities.initPlateVertexes(100, 100,0)
detectedTarget = np.zeros((4,2), dtype=np.float32)

runLoop = True

if builtinCamera:
    cap = cv2.VideoCapture(0)
else:
    cam = camera.Camera(4024,3036)
    cam.start()
    cv2.waitKey(500)
    frame = cam.latestFrame
    if frame is None:
        sys.exit(1)

# Lists of all circles and squares detected
circles = []
squares = []
targets = []

# The center of the license plate is:
#   (0,0,0) in world coordinates
#   (0,0,0,1) in homogeneneous world coordinates
plateCenter_Xw = np.zeros((4,1), dtype=np.float32)
plateCenter_Xw[3] = 1

M_K = load('./Support files/Camera/CameraMtx.npy')
M_DIST = load('./Support files/Camera/CameraDist.npy')
M_DInv = load('./Support files/Camera/M_DInv.npy')

while runLoop:

    if builtinCamera:
        _, image = cap.read()
        imageGray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    else:
        imageGray = cam.latestFrame
        if imageGray is None:
            sys.exit(1)

    _, imageBW = cv2.threshold(imageGray, 50, 255, 0)
    # Find all contours in the image
    contours, _ = cv2.findContours(imageBW, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    circles.clear()
    squares.clear()
    for contour in contours:
        approx = cv2.approxPolyDP(contour,0.01*cv2.arcLength(contour,True),True)
        area = cv2.contourArea(contour)
        if len(approx) == 4 and area > 1000:
            x, y, w, h = cv2.boundingRect(contour)
            ratio = float(w)/h
            if ratio >= 0.8 and ratio <= 1.2:
                square = [approx, x, y, w, h, area]
                squares.append(square)
        elif len(approx) >= 10 and cv2.isContourConvex(approx) and area > 100:
    	    M = cv2.moments(contour)
    	    x = int(round(M["m10"]/M["m00"]))
    	    y = int(round(M["m01"]/M["m00"]))
    	    circle = [contour, x, y, area]
    	    circles.append(circle)
    winner = None
    for square in squares:
        targets.clear()
        # If a circle is inside a square, it may belong to our target
        for circle in circles:
            if circle[1] > square[1] and circle[1] < square[1]+square[3] and circle[2] > square[2] and circle[2] < square[2]+square[4]:
                targets.append(circle)
        # If the square contains exactly 9 circles of the same area (within tolerances) it is definitely our target
        if len(targets) == 9:
            # Check if the 9 circles have the same area (within a given tolerance)
            area = 0
            for t in targets:
                area = area + t[3]
            area = area / len(targets)
            matchArea = True
            for t in targets:
                if t[3] < area * 0.8 or t[3] > area * 1.2:
                    matchArea = False
                    break
            if matchArea:
                if winner is None:
                    winner = square
                else:
                    if square[4] < winner[4]:
                        winner = square
                if drawCircles:
                    for t in targets:
                        cv2.circle(imageGray, (t[1],t[2]), 10, 255, -1)
    if winner is not None:
        i = 0
        for p in winner[0]:
            cv2.circle(imageGray,(p[0][0],p[0][1]),50,255,-1)
            detectedTarget[i][0] = p[0][0]
            detectedTarget[i][1] = p[0][1]
            i = i+1
        # Solve the projection problem by looking for the rotation
        # and translation vectors from word (3D) to screen (2D) coordinate
        _, rVec, tVec = cv2.solvePnP(realTarget, np.float32(detectedTarget), M_K, M_DIST, cv2.SOLVEPNP_P3P)
        M = Utilities.buildRototranslationMatrix(rVec, tVec)
        # By applying the rototranslation, we get its camera coordinates
        Mtot = np.matmul(M_DInv,M)
        plateCenter_Xd = np.matmul(Mtot, plateCenter_Xw)[:,0]
        plateCenter_Xd[2] *= -1
        #print("{:.0f} {:.0f} {:.0f}".format(plateCenter_Xd[0],plateCenter_Xd[1],plateCenter_Xd[2]))
        cv2.putText(imageGray, "{:.0f} {:.0f} {:.0f}".format(plateCenter_Xd[0],plateCenter_Xd[1],plateCenter_Xd[2]), (100,100), cv2.FONT_HERSHEY_SIMPLEX, 4, 255, 4, cv2.LINE_4)

        #cv2.drawContours(imageGray,[winner[0]],0,255,3)
        #cv2.drawContours(imageGray,[winner[0]],0,0,1)
    cv2.imshow('Window',imageGray)
    k = cv2.waitKey(10)
    if k == 27:
        runLoop = False

cam.close()
cv2.destroyAllWindows()




