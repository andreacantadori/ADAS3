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
    def __init__(self, serialNumber, width, height):
    #-----------------------------------------------------
        self.width = width
        self.height = height        
        for i in pylon.TlFactory.GetInstance().EnumerateDevices():
            if i.GetSerialNumber() == serialNumber:
                info = i
                break
        self.camera  = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateDevice(info))
        self.camera.Open()
        self.camera.Width.SetValue(self.width)
        self.camera.Height.SetValue(self.height)
        self.camera.CenterX.SetValue(True)
        self.camera.CenterY.SetValue(False)
        self.camera.OffsetY.SetValue(1012)
        self.camera.GainAuto.SetValue("Continuous")
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
