import cv2
import numpy as np
from time import time
import camera
import sys
import utilities
from numpy import load
from numpy import save
from numpy import asarray



calibrateXd = False

realTarget = utilities.initPlateVertexes(135,135,0)
detectedTarget = np.zeros((4,2), dtype=np.float32)

runLoop = True

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

p1 = np.zeros((4,1), dtype=np.float32)
p1[0] = -67
p1[1] = 0
p1[2] = 0
p1[3] = 1

M_K = load('./Support files/Camera/CameraMtx.npy')
print('M_K\n',M_K)
M_DIST = load('./Support files/Camera/CameraDist.npy')
print('M_DIST\n',M_DIST)
global M_DInv
M_DInv = load('./Support files/Camera/M_DInv.npy')
print('M_Dinv\n',M_DInv)

while runLoop:

    imageGray = cam.latestFrame
    if imageGray is None:
        sys.exit(1)

    _, imageBW = cv2.threshold(imageGray, 80, 255, 0)
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
        elif len(approx) >= 8 and cv2.isContourConvex(approx) and area > 1000:
            M = cv2.moments(contour)
            x = int(round(M["m10"]/M["m00"]))
            y = int(round(M["m01"]/M["m00"]))
            circle = [contour, x, y, area]
            circles.append(circle)
    winner = None
    for s in squares:
        targets.clear()
        # If a circle is inside a square, it may belong to our target
        for c in circles:
            if c[1] > s[1] and c[1] < s[1]+s[3] and c[2] > s[2] and c[2] < s[2]+s[4] and s[5] < c[3]*2.87*1.2 and s[5] > c[3]*2.87*0.8:
                targets.append(c)
        # If the square contains exactly 9 circles of the same area (within tolerances) it is definitely our target
        if len(targets) == 1:
            if winner is None:
                winner = s
            else:
                if s[4] < winner[4]:
                    winner = s
    if winner is not None:
        i = 0
        for p in winner[0]:
            detectedTarget[i][0] = p[0][0]
            detectedTarget[i][1] = p[0][1]
            i = i+1
        detectedTarget = utilities.orderPoints(detectedTarget)
        i = 0
        for p in detectedTarget:
            cv2.circle(imageGray,(p[0],p[1]),50,255,-1)
            cv2.putText(imageGray, "{:.0f}".format(i), (p[0],p[1]), cv2.FONT_HERSHEY_SIMPLEX, 4, 255, 4, cv2.LINE_4)
            i = i+1
        if calibrateXd:
            tmpTarget = utilities.initPlateVertexes(135,135,350)
            _, rVec, tVec = cv2.solvePnP(tmpTarget, np.float32(detectedTarget), M_K, M_DIST, cv2.SOLVEPNP_ITERATIVE)
            M = utilities.buildRototranslationMatrix(rVec, tVec)
            M_DInv = np.linalg.inv(M)
            save( './Support files/Camera/M_DInv', asarray(M_DInv))
            print("<<<< CALIBRATION MATRIX >>>")
            print(M_DInv)
            calibrateXd = False
        else:
            # Solve the projection problem by looking for the rotation
            # and translation vectors from word (3D) to screen (2D) coordinate
            _, rVec, tVec = cv2.solvePnP(realTarget, np.float32(detectedTarget), M_K, M_DIST, cv2.SOLVEPNP_P3P)
            M = utilities.buildRototranslationMatrix(rVec, tVec)
            centerTarget = np.zeros((1,3), np.float32)
            projectedTarget, _ =  cv2.projectPoints(centerTarget, rVec, tVec, M_K, M_DIST) 
            for p in projectedTarget:
                cv2.circle(imageGray,(int(p[0][0]),int(p[0][1])),30,0,-1)

            # By applying the rototranslation, we get its camera coordinates
            #Mtot = np.matmul(M_DInv,M)
            #plateCenter_Xd = np.matmul(Mtot, plateCenter_Xw)[:,0]
            #plateCenter_Xd[2] *= -1
            #cv2.putText(imageGray, "{:.0f} {:.0f} {:.0f}".format(plateCenter_Xd[0],plateCenter_Xd[1],plateCenter_Xd[2]), (100,100), cv2.FONT_HERSHEY_SIMPLEX, 4, 255, 4, cv2.LINE_4)               
            print(tVec)
            cv2.putText(imageGray, "{:.0f} {:.0f} {:.0f}".format(tVec[0][0],tVec[1][0],tVec[2][0]), (100,100), cv2.FONT_HERSHEY_SIMPLEX, 4, 255, 4, cv2.LINE_4)
    cv2.line(imageGray,(0,1518),(4024,1518),255,3)
    cv2.line(imageGray,(2012,0),(2012,3036),255,3)
    cv2.imshow('Window',imageGray)
    

    
    k = cv2.waitKey(10)
    if k == utilities.Key.ESC:
        runLoop = False

cam.close()
cv2.destroyAllWindows()




