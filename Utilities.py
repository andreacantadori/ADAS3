class Key:
    # ASCII codes
    ESC   = 27
    SPACE = 32
    R     = 82
    r     = 114
    
#====================================================
# Graphic utilities
#====================================================
import cv2
import numpy as np
from scipy.spatial import distance as dist

#-------------------------------------------------
def orderPoints(pts):
#-------------------------------------------------
    # sort the points based on their x-coordinates
    xSorted = pts[np.argsort(pts[:, 0]), :]
    # grab the left-most and right-most points from the sorted x-roodinate points
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]
    # now, sort the left-most coordinates according to their y-coordinates
    # so we can grab the top-left and bottom-left points, respectively
    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (topLeft, bottomLeft) = leftMost
    # now that we have the top-left coordinate, use it as an
    # anchor to calculate the Euclidean distance between the
    # top-left and right-most points; by the Pythagora
    # theorem, the point with the largest distance will be
    # our bottom-right point
    D = dist.cdist(topLeft[np.newaxis], rightMost, "euclidean")[0]
    (bottomRight, topRight) = rightMost[np.argsort(D)[::-1], :]
    # return the coordinates in top-left, top-right,
    # bottom-right, and bottom-left order
    return np.intp([topLeft, topRight, bottomRight, bottomLeft])

#----------------------------------------------------
def buildRototranslationMatrix(r, t):
# Given a rotation matrix in Rodrigues notation
# and a translation vector, builds and returns
# the rototranslation matrix
#----------------------------------------------------
    rMatrix, _ = cv2.Rodrigues(r)
    # Now fill the rototranslation matrix in:
    m = np.zeros((4,4), dtype=np.float32)
    m[0][0] = rMatrix[0][0]
    m[0][1] = rMatrix[0][1]
    m[0][2] = rMatrix[0][2]
    m[0][3] = t[0]
    m[1][0] = rMatrix[1][0]
    m[1][1] = rMatrix[1][1]
    m[1][2] = rMatrix[1][2]
    m[1][3] = t[1]
    m[2][0] = rMatrix[2][0]
    m[2][1] = rMatrix[2][1]
    m[2][2] = rMatrix[2][2]
    m[2][3] = t[2]
    m[3][3] = 1
    return m

#----------------------------------------------------
def roi2FrameCoordinates(rect, roi):
# Given a ROI within the screen frame, and a
# rectangle rect in ROI coordinates, returns 
# the rectangle in screen coordinates
# Inputs:
#   rect: rectangle in ROI coordinates
#   roi: ROI in screen coordinates
#----------------------------------------------------
    rect[0][0] += roi[0][0]
    rect[1][0] += roi[0][0]
    rect[2][0] += roi[0][0]
    rect[3][0] += roi[0][0]
    rect[0][1] += roi[0][1]
    rect[1][1] += roi[0][1]
    rect[2][1] += roi[0][1]
    rect[3][1] += roi[0][1]
    return rect
    
#----------------------------------------------------
def initPlateVertexes(w, h, d):
# Returns the vertexes of the license plate in license plate coordinates
# starting from the top left point and proceding clockwise
#----------------------------------------------------
    m = np.zeros((4,3), np.float32)
    
    m[0][0] = - w / 2
    m[0][1] =   h / 2
    m[0][2] =   d 
    
    m[1][0] =   w / 2 
    m[1][1] =   h / 2 
    m[1][2] =   d 
    
    m[2][0] =   w / 2 
    m[2][1] = - h / 2 
    m[2][2] =   d 
    
    m[3][0] = - w / 2 
    m[3][1] = - h / 2 
    m[3][2] =   d     
    
    return m

