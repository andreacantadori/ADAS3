import numpy as np
from numpy import save
from numpy import load
from numpy import asarray
import cv2
import glob
import time
from pypylon import pylon
import camera

# ----------------------------------------------------------------------
def takeChessboardPictures():
# ----------------------------------------------------------------------

    globalWidth = 4024
    globalHeight = 2024 #3036
    # Create camera object (encapsulates pypylon) and set the camera to its maximum resolution
    cam = camera.Camera('24686187',globalWidth,globalHeight)

    result, frame = cam.getFrame()
    cv2.namedWindow( 'Chessboard acquisition' )
    cv2.imshow( 'Chessboard acquisition', frame )

    exitLoop = False
    nPicture = 0
    time0 = time.time()
    cnt = 10
    while not exitLoop:
        result, frame = cam.getFrame()
        cv2.imshow( 'Chessboard acquisition', frame )
        if time.time()-time0 > 1:
            time0 = time.time()
            cnt -= 1
            for i in range(cnt):
                print( "#",end='' )
            print()
            if cnt == 1:
                print( "<=========================" )
            elif cnt == 0:
                result, frame = cam.getFrame()
                cv2.imshow( 'Chessboard acquisition', frame )
                nPicture += 1
                fName = 'Chessboard{}.png'.format( nPicture )
                cv2.imwrite( './ChessboardPhotos/' + fName, frame)
                print( fName )
                cnt = 10
        k = cv2.waitKey( 10 ) % 256
        if k == 27:
            exitLoop = True
    cam.close()
    cv2.destroyAllWindows()
    
# ----------------------------------------------------------------------
def calibrateCameraWithChessboard( nRows, nCols, squareSize ):
# ----------------------------------------------------------------------

    cv2.namedWindow( 'Camera calibration' )

    terminationCriteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Prepare object points in world coordinates, like (0,0,0) (1,0,0) ... (7,7,0)
    objectPoints = np.zeros( ( nRows * nCols, 3 ), np.float32 )
    objectPoints[:,:2] = np.mgrid[0:nRows,0:nCols].T.reshape(-1,2)
    objectPoints *= squareSize
    # Arrays to store object points and image points from all the images.
    worldPoints = [] # 3d point in real world space
    imgPoints = []   # 2d points in image plane.


    # Calibration...
    images = glob.glob( './ChessboardPhotos/Chessboard*.png' )
    for fName in images:
        
        # Get a chessboard image and convert to grayscale
        print( fName )
        img = cv2.imread( fName )
        imgGray = cv2.cvtColor( img, cv2.COLOR_BGR2GRAY )
        
        # Find the chessboard corners
#        ret, corners = cv2.findChessboardCorners( imgGray, ( nRows, nCols ), None )
        ret, corners = cv2.findChessboardCorners(
                imgGray, (nRows,nCols),
                cv2.CALIB_CB_ADAPTIVE_THRESH
                + cv2.CALIB_CB_FAST_CHECK +
                cv2.CALIB_CB_NORMALIZE_IMAGE)

        
        # If found, add object points and image points (after refining them)
        if ret == True:
            worldPoints.append( objectPoints )
            corners2 = cv2.cornerSubPix( imgGray, corners, (11,11), (-1,-1), terminationCriteria )
            imgPoints.append( corners2 )
            # Draw and display the corners
            cv2.drawChessboardCorners( img, (nRows,nCols), corners2, True )
            cv2.imshow( 'Camera calibration', img )
            cv2.waitKey( 4000 )

    print("==========================")
    _, mtx, dist, rvecs, tvecs = cv2.calibrateCamera( worldPoints, imgPoints, imgGray.shape[::-1], None, terminationCriteria )
    save( './Support files/Camera/CameraMtx', asarray( mtx ) )
    save( './Support files/Camera/CameraDist', asarray( dist ) )
    print( mtx )
    print( dist )

    mean_error = 0
    for i in range( len( worldPoints ) ):
        imgPoints2, _ = cv2.projectPoints( worldPoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm( imgPoints[i], imgPoints2, cv2.NORM_L2 ) / len( imgPoints2 )
        mean_error += error
    print( "total error: {}".format(mean_error/len( worldPoints ) ) )

    cv2.destroyAllWindows()


#takeChessboardPictures()
calibrateCameraWithChessboard( nRows = 7, nCols = 10, squareSize = 37 )

