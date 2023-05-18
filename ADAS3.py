#!/usr/bin/python3

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
def getTargetVertexesInClockwiseOrder(t):
#--------------------------------------------------------------------------------------------------
    pts = np.zeros((4,2), dtype=np.float32)
    i = 0
    for p in t[0]:
        pts[i,0] = p[0,0]
        pts[i,1] = p[0,1]
        i = i+1
    return utilities.orderPoints(pts)

#--------------------------------------------------------------------------------------------------
def isCircleInscribedInSquare(circle, square):
    w = square[3]
    h = square[4]
    sX = square[1] + int(w/2)
    sY = square[2] + int(h/2)
    l = int((w+h)/2)
    d = circle[4] * 2    # Diameter
    x = circle[1]
    y = circle[2]
    # Check if the diameter of the circle is equal to 0.8 times the side of the square (+/- a tolerance)
    # and check if the center of the circle is "quite" close to the center of the square
    if d<l*0.8*1.1 and d>l*0.8*0.9 and abs(x-sX)<100 and abs(y-sY)<100:
        res = True
    else:
        res = False
    return res    

#--------------------------------------------------------------------------------------------------
def showVertexes(img, target, show):
#--------------------------------------------------------------------------------------------------
    if show:
        i = 0
        for p in target:
            cv2.circle(img,(p[0],p[1]),5,255,-1)
            cv2.putText(img, "{:.0f}".format(i), (p[0],p[1]), cv2.FONT_HERSHEY_SIMPLEX, 2, 255, 4, cv2.LINE_4)
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
def showCoordinates(img, coord, pos, sbp, show):
#--------------------------------------------------------------------------------------------------
    if show:
        if sbp:
            cv2.putText(img, "X,Y,Z (mm): {:.0f},{:.0f},{:.0f}, subpixel".format(coord[0],coord[1],coord[2]), (pos[0],pos[1]), cv2.FONT_HERSHEY_SIMPLEX, 4, 255, 4, cv2.LINE_4)
        else:
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

#--------------------------------------------------------------------------------------------------
def subpixelAnalysis(img,corners):
#--------------------------------------------------------------------------------------------------
    # Set the needed parameters to find the refined corners
    winSize = (5, 5)
    zeroZone = (-1, -1)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TermCriteria_COUNT, 40, 0.001)
    # Convert corners (4x2 matrix) to expected input format (4x1x2 matrix) - this is required by Numpy libs
    c = np.zeros((4,1,2), np.float32)
    for i in range(4):
        c[i,0,0] = corners[i,0]
        c[i,0,1] = corners[i,1]
    # Calculate the refined corner locations
    c = np.int16(cv2.cornerSubPix(img, c, winSize, zeroZone, criteria))
    # Switching back to the original (4x2) format
    result = np.zeros((4,2),np.int16)
    for i in range(4):
        result[i,0] = c[i,0,0]
        result[i,1] = c[i,0,1]
    return result


#--------------------------------------------------------------------------------------------------
# MAIN PROGRAM STARTS HERE
#--------------------------------------------------------------------------------------------------
def main():
    # Graphic options
    graphicsShowVertexes = False    # Set to True to show vertexes
    graphicShowCross = False        # Call with True to display the white cross centered in the image

    # Load camera intrinsic matrix
    M_K = load('./Support files/Camera/CameraMtx.npy')
    # Load camera distortion matrix
    M_DIST = load('./Support files/Camera/CameraDist.npy')
    # ========>>>> TODO: Manage the cases of missing matrixes!

    # The real (i.e. in world coordinates) is a square of 200x200 mm
    # ========>>>> ADJUST THESE SIZES AS APPROPRIATE!
    # Note that the z-coordinate in the real target's coordinate system is 0
    # See buildRealTarget for more details
    realTarget = utilities.buildRealTarget(200,200,0)

    # The detected target, as displayed on the 2D screen, is an array of 4 (x,y) points
    detectedTarget = np.zeros((4,2), dtype=np.float32)
    # The complete target consists of the 4 vertexes of the square plus the center of the detected circle
    # It is therefore an array of 5 (x,y) points
    # The addition of the circle center adds significant more stability to the projection algorithm (solvePnP):
    # in fact, if the camera is not exactly focused on the target (which is impossible), the contour of the square is not a clear B/W transition,
    # but a gray shaded area; because of binarization, the size of this area depends on the threshold (at least to some extent): in other
    # words, the solution of the solvePnP algorithm slightly depends on the binarization threshold, too.
    # By adding a 5-th point, i.e. the center of the circle (which is invariant to the choosen threshold, at least at first approximation),
    # the solution of the projection algorithm is less affected by the threshold
    extendedTarget = np.zeros((5,2), dtype=np.float32)

    # OUTPUT CUSTOMIZAZION
    # Enable or disable subpixel analysis
    subpixelAnalysysEnabled = True
    # Enable or disable display of target vertexes
    showVertexesEnabled = True
    # Enable or disable coordinate printout as image overlay
    showCoordinatesEnable = True
    # Enable or disable lowpass filtering of (x,y,z) coordinagtes
    filterCoordinatesEnable = True
    # (x,y,z) coordinates low pass filter constant
    # Just ignore if filterCoordinatesEnable is False
    alfa = 0.1

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

    nFrame = 0
    xchgFile = open('xcghFile.txt','w')
    xchgFile.close()

    avgX = 0
    avgY = 0 
    avgZ = 0
    avgRoll = 0
    avgPitch = 0 
    avgYaw = 0

    print('---------------------------------')
    print('              ADAS3')
    print('---------------------------------')
    print('Interactive line commands:')
    print('     V/v: enable/disable display of target vertexes')
    print('     S/s: enable/disable subpixel analysis')
    print('     ESC: exit program')

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))

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
        contours, _ = cv2.findContours(imageBW, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        # Some contours may be square, others may be circles: get all of them into
        # their respective lists
        circles.clear()
        squares.clear()
        for contour in contours:
            # Approximate the contour with a polyline and get its area (in pixels)
            approx = cv2.approxPolyDP(contour,0.01*cv2.arcLength(contour,True),True)
            area = cv2.contourArea(contour)
            # A contour is a square if it is approximated by a polyline with 4 segments
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
                # A circle was detected: let's find its center and radius
                (x,y),radius = cv2.minEnclosingCircle(approx)
                # Append the circle and its properties to the list
                circles.append([approx, x, y, area, radius])
        maybeTarget = None
        circleCenter = None
        for s in squares:
            targets.clear()
            # If a circle lays inside a square, and its area is 2.87 (+/- a tolerance) times 
            # smaller than that of the square, then that square is a candidate to be our target
            for c in circles:
                if isCircleInscribedInSquare(c,s):
                    targets.append(c)
            # If the square contains exactly 1 circle, it is candidate maybeTarget
            if len(targets) == 1:
                if maybeTarget is None:
                    maybeTarget = s
                else:
                    # If a maybeTarget was already nominated, and the new candidate has smaller area than it,
                    # then the new candidate becomes the maybeTarget
                    # In other words: we choose the smallest square which contains one and only one circle
                    if s[4] < maybeTarget[4]:
                        maybeTarget = s
                circleCenter = (targets[0][1],targets[0][2])
                # Adjust binary threshold until the ratio between square and circle area is 1.99
                # If light threshold increases, the square area decreases and the circle area increases
                # If light threshold decreases, the square area increases and the circle area decreases
                if c[3]*1.99 > s[5]:
                #if c[4]*2.50 > s[3]:
                    if lightThreshold > 0:
                        lightThreshold = lightThreshold-1
                else:
                    if lightThreshold < 255:
                        lightThreshold = lightThreshold+1
        if maybeTarget is None:
            # We could not find any maybeTarget, i.e. any target
            # Therefore we adjust our binary threshold by lowering it by 10% and retry...
            # Once the threshold is too low, we restart from the beginning
            lightThreshold = int(lightThreshold * 0.9)
            if lightThreshold <= 1:
                lightThreshold = None
                utilities.sharePosition(nFrame,None,None,None,None,None,None)
        else:
            # Finally we have a maybeTarget: our target in the image!
            # The maybeTarget is a square, defined by its 4 vertexes. For the projection algorithm to work properly,
            # we need that the vertexes of the square are ordered always the same way, e.g. the first being the 
            # top left, the second the top right, and so on moving clockwise. 
            # This is what is done by the orderPoints() method
            detectedTarget = getTargetVertexesInClockwiseOrder(maybeTarget)
            if subpixelAnalysysEnabled:
                detectedTarget = subpixelAnalysis(imageGray,detectedTarget)
            # Call showVertexes(_,True) to highlight the vertexes in the image
            imageGray = showVertexes(imageGray, detectedTarget, showVertexesEnabled) 
            # Solve the projection problem by looking for the rotation
            # and translation vectors from word (3D) to screen (2D) coordinate
            extendedTarget[0] = detectedTarget[0]
            extendedTarget[1] = detectedTarget[1]
            extendedTarget[2] = detectedTarget[2]
            extendedTarget[3] = detectedTarget[3]
            extendedTarget[4,0] = circleCenter[0]
            extendedTarget[4,1] = circleCenter[1]
            _, rVec, tVec = cv2.solvePnP(realTarget, np.float32(extendedTarget), M_K, M_DIST, cv2.SOLVEPNP_P3P)
            # rVec is the (4x1) rotation vector (quaternion)
            # tVec is the (3x1) translation vector, corresponding to the displacement of the center
            # of the real target in the camera coordinate system
            x = tVec[0,0]
            y = tVec[1,0]
            z = tVec[2,0]
            # Some lowpass filtering...
            if filterCoordinatesEnable:
                avgX = x * alfa + (1-alfa) * avgX
                avgY = y * alfa + (1-alfa) * avgY
                avgZ = z * alfa + (1-alfa) * avgZ
            else:
                avgX = x
                avgY = y
                avgZ = z
            showCoordinates(imageGray, (avgX,avgY,avgZ), (100,100), subpixelAnalysysEnabled, showCoordinatesEnable)
            # Calculate the pitch, yaw, and roll angles
            # Pitch is rotation around x-axis, yaw is rotation around y-axis, and roll is rotation around z-axis
            # Step 1: convert the quaternion to a (3x3) rotation matrix
            rotMat = cv2.Rodrigues(rVec)[0]
            # Step 2: calculate roll, pitch, yaw from rotation matrix
            roll_rad = np.arctan2(rotMat[1, 0], rotMat[0, 0])
            pitch_rad = np.arctan2(rotMat[2, 1], rotMat[2, 2])
            yaw_rad = np.arctan2(-rotMat[2, 0], np.sqrt(rotMat[2, 1]**2 + rotMat[2, 2]**2))
            # Convert the angles from radians to degrees
            roll_deg = np.rad2deg(roll_rad)
            pitch_deg = np.rad2deg(pitch_rad)
            if pitch_deg > 0:
                pitch_deg = 180 - pitch_deg
            else:
                pitch_deg = -pitch_deg -180
            yaw_deg = np.rad2deg(yaw_rad)
            # Some lowpass filtering...
            avgRoll = roll_deg * alfa + (1-alfa) * avgRoll
            avgYaw = yaw_deg * alfa + (1-alfa) * avgYaw
            avgPitch = pitch_deg * alfa + (1-alfa) * avgPitch
            showAngles(imageGray, (avgRoll,avgPitch,avgYaw), (100,250), True)
            # Output real-time position information for other apps
            utilities.sharePosition(nFrame,avgX,avgY,avgZ,avgRoll,avgPitch,avgYaw)
        
        # Graphic output... 
        showCross(imageGray, graphicShowCross) 
        cv2.imshow('Window',imageGray)
        nFrame = nFrame + 1
        k = cv2.waitKey(10)
        if k == utilities.Key.ESC:
            runLoop = False
        elif k == utilities.Key.S:
            subpixelAnalysysEnabled = True
            print('>>> Subpixel analysis enabled')
        elif k == utilities.Key.s:
            subpixelAnalysysEnabled = False
            print('>>> Subpixel analysis disabled')
        elif k == utilities.Key.V:
            showVertexesEnabled = True
            print('>>> Show vertexes enabled')
        elif k == utilities.Key.T:
            lightThreshold = lightThreshold + 1
            print('lightThreshold', lightThreshold)
        elif k == utilities.Key.t:
            lightThreshold = lightThreshold - 1
            print('lightThreshold', lightThreshold)
        elif k == utilities.Key.v:
            showVertexesEnabled = False
            print('>>> Show vertexes disabled')


    # Close everything and shut down
    cam.stop()
    cv2.waitKey(1000)
    cam.close()
    cv2.destroyAllWindows()

#------------------------------------------------------------------------
#                            MAIN
#------------------------------------------------------------------------
if __name__ == "__main__":
    for i in range(1, len(sys.argv)):
        if sys.argv[i] == '-calibrate' or sys.argv[i] == '-c':
            calibrateXd = True
            print(">>> Calibration mode")
    main()

