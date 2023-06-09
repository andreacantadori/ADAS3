import cv2
import numpy as np
from pypylon import pylon
from threading import Thread
import time

#-----------------------------------------------------
# Camera class
# Reads the video frame from a Pylon-compatible camera
# (e.g. the Basler ac4024)
# The camera process runs in a thread
# The most recent frame is accessible via self.latestFrame
#-----------------------------------------------------
class Camera:

    #-----------------------------------------------------
    def __init__(self, width, height):
    #-----------------------------------------------------
        self.width = width
        self.height = height
        self.camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
        self.camera.Open()
        self.camera.Width.SetValue(self.width)
        self.camera.Height.SetValue(self.height)
        self.camera.CenterX.SetValue(True)
        self.camera.CenterY.SetValue(True)
        # =======>>>>> TODO: set the automatic gain mode
        # ... I could not find the corresponding command... The only way so far is to use Pylon Viewer...
        # self.camera.GainAuto.SetValue(GainAuto_Continuous)
        self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        self.converter = pylon.ImageFormatConverter()
        self.converter.OutputPixelFormat = pylon.PixelType_Mono8
        self.converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
        self.stopped = False
        
    #-----------------------------------------------------
    def getFrame(self):
    #-----------------------------------------------------
        grabResult = self.camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
        if grabResult.GrabSucceeded():
            convertedGrab = self.converter.Convert(grabResult)
            frame = convertedGrab.GetArray()
            return True, frame
        else:
            return False, None
    
    #-----------------------------------------------------
    def close(self):
    #-----------------------------------------------------
        self.camera.StopGrabbing()
        self.camera.Close()
        self.latestFrame = None

    #-----------------------------------------------------
    def start(self):
    #-----------------------------------------------------
        self.stopped = False
        Thread(target = self.getFrameAsync, args = ()).start()
        return self
                
    #-----------------------------------------------------
    def getFrameAsync(self):
    #-----------------------------------------------------
        while not self.stopped:
            time.sleep(0.01)
            grabResult = self.camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
            if grabResult.GrabSucceeded():
                convertedGrab = self.converter.Convert(grabResult)
                frame = convertedGrab.GetArray()
                self.latestFrame = frame
    
    #-----------------------------------------------------
    def stop(self):
    #-----------------------------------------------------
        self.stopped = True
