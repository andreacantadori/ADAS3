import cv2
import numpy as np
from time import time
from operator import itemgetter

# Set to True to see step-by-step image processing
showIntermediate = False

imagePath = "SceneB1.png"
image = cv2.imread(imagePath)
imageCopy = image.copy()
time0 = time()
# Convert to grayscale
imageGray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
if showIntermediate:
    cv2.imshow('Win',imageGray)
    cv2.waitKey(0)

# Find all contours in the image
ret, edgeImage = cv2.threshold(imageGray, 127, 255, 0)
if showIntermediate:
    cv2.imshow('Win',edgeImage)
    cv2.waitKey(0)

circles = []
squares = []

contours, _ = cv2.findContours(edgeImage, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
imgCopy = image.copy()
cv2.drawContours(imgCopy, contours, -1, (0,0,255), 3)
if showIntermediate:
    cv2.imshow('Win',imgCopy)
    cv2.waitKey(0)

for contour in contours:
    approx = cv2.approxPolyDP(contour,0.01*cv2.arcLength(contour,True),True)
    area = cv2.contourArea(contour)
    if len(approx) == 4 and area > 100:
        x, y, w, h = cv2.boundingRect(contour)
        ratio = float(w)/h
        if ratio >= 0.9 and ratio <= 1.1:
            square = [contour, x, y, w, h, area]
            squares.append(square)
    elif ((len(approx) > 10) & (area > 1000) ):
    	M = cv2.moments(contour)
    	x = int(round(M["m10"]/M["m00"]))
    	y = int(round(M["m01"]/M["m00"]))
    	circle = [contour, x, y, area]
    	circles.append(circle)

for square in squares:
    targets = []
    for circle in circles:
        if circle[1] > square[1] and circle[1] < square[1]+square[3] and circle[2] > square[2] and circle[2] < square[2]+square[4]:
            targets.append(circle)
    if len(targets) == 9:
        # Check if the 9 circles have the same area (within a given tolerance)
        area = 0
        for t in targets:
            area = area + t[3]
        area = area / len(targets)
        matchArea = True
        for t in targets:
            if t[3] < area * 0.9 or t[3] > area * 1.1:
                matchArea = False
                break
        if matchArea:
            # Check if they lay down in a regular grid...
            for s in targets:
                print(s[1],s[2],s[3])
            for t in targets:
                cv2.circle(image, (t[1],t[2]), 10, (0,0,255), -1)
            break
                
print(time()-time0)

cv2.imshow('Window',image)
cv2.waitKey(0)
quit()



