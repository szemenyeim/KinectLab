import cv2
import numpy as np
from BBGUI import BBGUI
import matplotlib.pyplot as plt

class ImageProcessor(object):
    def __init__(self):

        self.BBGUI = BBGUI()

        self.focal = 1062.3
        self.imgW = 640
        self.imgH = 480

        self.initDepth = None
        self.initW = None
        self.initH = None

        self.BB = None
        self.cog = None

        self.depthVal = None
        self.width = None
        self.height = None
        self.hueVal = None

    def updateBB(self,cog, BBSize, imSize):

        # Half width and half heigth
        hWidth = BBSize[0] // 2
        hHeigth = BBSize[1] // 2

        # Check if the new bb is within the image boundary
        if cog[0] <= hWidth:
            cog = (hWidth + 1, cog[1])
        if cog[1] <= hHeigth:
            cog = (cog[0], hHeigth + 1)

        if cog[0] + hWidth > imSize[1]:
            hWidth = imSize[1] - cog[0]
        if cog[1] + hHeigth > imSize[0]:
            hHeigth = imSize[0] - cog[1]

        return (cog[1] - hHeigth, cog[1] + hHeigth, cog[0] - hWidth, cog[0] + hWidth)

    def getRoi(self,array, ROI):
        # Cut ROI from array
        return None

    def getDepthMask(self,depthRoi):
        # get the depth value in the middle of the depth ROI (CONVERT IT TO INT!!!)


        # If the jump from the previous value is too large, use the previous value


        # If there is nothing in the initDepth variable, fill it with the current depthVal


        # Mask the depth image using the current depthVal

        return np.ones(depthRoi.shape, 'uin8')

    def getLargestMask(self, binary):

        # Initialize outputs
        mask = np.zeros_like(binary)
        moments = None

        # Find contours

        # get largest contour

        # In case of no contours return default ones

        # If getMoments is true, compute moments

        # Draw largest contour on image and return
        return mask, moments

    def getMaxHue(self, image, mask):
        # Mask hsv image TIP: use cv2.bitwise_and

        # Get hue, saturation and value images

        # Mask out hue pixels with 0 value and saturation smaller thatn 20

        # compute histogram

        # return argmax of histogram
        return 0

    def getHueMask(self,imgRoi,depthMask):

        # Convert image to HSV


        # If there is nothing in hueVal, get the largest hue


        # Mask the hsv image using the hueVal. Also use saturation and value

        return np.ones(imgRoi.shape, 'uin8')

    def processImage(self, image, depth):
        if self.BB is None:
            self.BB = self.BBGUI.getBB(image)
        # self.BB has [y1,y2,x1,x2]

        # Set width and height (including initial)


        # get ROIs


        # Get depth mask


        # Get hue mask


        # combine the two masks


        # Find largest contour in final mask


        # If the moments were not found, return images


        # Compute new center of mass (DON'T FORGET TO COMPENSATE FOR THE ROI!!!)


        # Draw the center of mass and bounding box on the image


        # Compute ratio between the original and current depths


        # Compute new size of the bounding rect using the ratio


        # Compute new bounding box

        # Return images
        return image, depth

    def compute3D(self):
        # Compute 3D coordinates
        # HELP: u = f/z*x; v = f/z*y
        z = -1
        x = -1
        y = -1
        return x,y,z

