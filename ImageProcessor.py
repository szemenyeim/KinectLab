import cv2
import numpy as np
from BBGUI import BBGUI
import matplotlib.pyplot as plt

class ImageProcessor(object):
    def __init__(self):

        self.BBGUI = BBGUI()

        self.focal = 1062.3
        self.imgW = 640
        self.imgH = 360

        self.initDepth = None
        self.initW = None
        self.initH = None

        self.hueVal = None

        self.BB = None
        self.cog = None

        self.depthVal = None
        self.width = None
        self.height = None

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
        # todo: Cut ROI from array
        return None

    def getDepthMask(self,depthRoi):
        # todo: get the depth value in the middle of the depth ROI (CONVERT IT TO INT!!!)


        # todo: If the jump from the previous value is too large, use the previous value


        # If there is nothing in the initDepth variable, fill it with the current depthVal
        if self.initDepth is None:
            self.initDepth = self.depthVal


        # todo: Mask the depth image using the current depthVal

        return np.ones(depthRoi.shape, 'uint8')

    def getLargestMask(self, binary):

        # Initialize outputs
        mask = np.zeros_like(binary)
        moments = None

        # todo: Find contours

        # todo: get largest contour

        # todo: In case of no contours return default ones

        # todo: If getMoments is true, compute moments

        # Draw largest contour on image and return
        return mask, moments

    def getMaxHue(self, image, mask):
        # todo: Mask hsv image TIP: use cv2.bitwise_and

        # todo: Get hue, saturation and value images

        # todo: Mask out hue pixels with 0 value and saturation smaller thatn 20

        # todo: compute histogram

        # todo: return argmax of histogram
        return 0

    def getHueMask(self,imgRoi,depthMask):

        # todo: Convert image to HSV


        # todo: If there is nothing in hueVal, get the largest hue


        # todo: Mask the hsv image using the hueVal. Also use saturation and value

        return np.ones(depthMask.shape, 'uint8')

    def processImage(self, image, depth):
        if self.BB is None:
            self.BB = self.BBGUI.getBB(image)
        # self.BB has [y1,y2,x1,x2]

        # Set width and height (including initial)
        self.width = self.BB[3] - self.BB[2]
        self.height = self.BB[1] - self.BB[0]
        if self.initH is None:
            self.initH = self.height
            self.initW = self.width

        # todo: get ROIs


        # todo: Get depth mask


        # todo: Get hue mask


        # todo: combine the two masks


        # todo: Find largest contour in final mask


        # todo: If the moments were not found, return images


        # todo: Compute new center of mass (DON'T FORGET TO COMPENSATE FOR THE ROI!!!)
        # cog has (x,y) coordinates


        # todo: Compute ratio between the original and current depths


        # todo: Compute new size of the bounding rect using the ratio


        # todo: Compute new bounding box


        # Draw the center of mass and bounding box on the image
        cv2.circle(image, self.cog, 10, (0, 0, 255), 2)
        cv2.circle(depth, self.cog, 10, (65535), 2)
        cv2.rectangle(image, (self.BB[2], self.BB[0]), (self.BB[3], self.BB[1]), (0, 0, 255), 1)
        cv2.rectangle(depth, (self.BB[2], self.BB[0]), (self.BB[3], self.BB[1]), (65535), 1)

        # Return images
        return image, depth

    def compute3D(self):
        # todo: Compute 3D coordinates
        # HELP: u = f/z*x + px; v = f/z*y + py
        z = -1
        x = -1
        y = -1
        return x,y,z

