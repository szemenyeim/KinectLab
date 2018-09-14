import cv2
import numpy as np
from BBGUI import BBGUI
import matplotlib.pyplot as plt

def updateBB(cog, BBSize, imSize):

    # Half width and half heigth
    hWidth = BBSize[0]//2
    hHeigth = BBSize[1]//2

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

def getRoi(array, ROI):
    return array[ROI[0]:ROI[1],ROI[2]:ROI[3]]

def getLargestMask(binary,getMoments=False):

    # Initialize outputs
    mask = np.zeros_like(binary)
    moments = None

    # Find contours
    _, contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # get largest contour
    maxSize = -1
    maxInd = -1
    for i, cont in enumerate(contours):
        area = cv2.contourArea(cont)
        if area > maxSize:
            maxSize = area
            maxInd = i

    # In case of no contours return default ones
    if maxInd < 0 or maxSize < 1:
        return mask, moments

    # If getMoments is true, compute moments
    if getMoments:
        moments = cv2.moments(contours[maxInd])

    # Draw largest contour on image and return
    return cv2.drawContours(mask, contours, maxInd, 255, -1), moments

def getMaxHue(image, mask):
    # Mask hsv image TIP: use cv2.bitwise_and
    maskedImgHsv = cv2.bitwise_and(image, image, mask=mask)

    # Get hue, saturation and value images
    hueImg = maskedImgHsv[:, :, 0]
    satImg = maskedImgHsv[:, :, 1]
    valImg = maskedImgHsv[:, :, 2]

    # Mask out hue pixels with 0 value and saturation smaller thatn 20
    hueMask = np.logical_and(valImg != 0, satImg > 20)
    hueVals = hueImg[hueMask]

    # compute histogram
    h = np.histogram(hueVals, 179)[0]

    # return argmax of histogram
    return np.argmax(h)

class ImageProcessor(object):
    def __init__(self):

        self.BB = None
        self.depthVal = None
        self.hueVal = None
        self.BBGUI = BBGUI()
        self.cog = None

        self.focal = 1062.3
        self.imgW = 640
        self.imgH = 480

        self.initDepth = None
        self.initW = None
        self.initH = None

    def processImage(self, image, depth):
        if self.BB is None:
            self.BB = self.BBGUI.getBB(image)

        # Set width and height
        width = self.BB[3]-self.BB[2]
        height = self.BB[1]-self.BB[0]
        if self.initH is None:
            self.initH = height
            self.initW = width

        # get ROIs
        imgRoi = getRoi(image,self.BB)
        depthRoi = getRoi(depth,self.BB)

        # get the depth value in the middle of the depth ROI (CONVERT IT TO INT!!!)
        midDepth = int(depthRoi[height//2,width//2])

        # If the jump from the previous value is too large, use the previous value
        if self.depthVal is not None and abs(midDepth - self.depthVal) > 250:
            midDepth = self.depthVal
        self.depthVal = midDepth

        # If there is nothing in the initDepth variable, fill it with the current depthVal
        if self.initDepth is None:
            self.initDepth = self.depthVal

        # Mask the depth image using the current depthVal
        roiMask = cv2.inRange(depthRoi, np.array([self.depthVal-100]), np.array([self.depthVal+100]))

        # Get largest contour mask
        roiMask,_ = getLargestMask(roiMask)

        # Convert image to HSV
        imgHsv = cv2.cvtColor(imgRoi,cv2.COLOR_BGR2HSV)

        # If there is nothing in hueVal, get the largest hue
        if self.hueVal is None:
            self.hueVal = getMaxHue(imgHsv,roiMask)

        # Mask the hsv image using the hueVal. Also use saturation and value
        hueMask = cv2.inRange(imgHsv,lowerb=np.array([self.hueVal-25, 10, 1]),upperb=np.array([self.hueVal+25, 255, 255]))

        # combine the two masks
        finalMask = np.bitwise_and(roiMask,hueMask)

        # Find largest contour in final mask
        finalMask, moments = getLargestMask(finalMask, getMoments=True)

        # If the moments were not found, return images
        if moments is None:
            return image, depth

        # Compute new center of mass (DON'T FORGET TO COMPENSATE FOR THE ROI!!!)
        self.cog = (int(moments['m10']/moments['m00']) + self.BB[2],int(moments['m01']/moments['m00']) + self.BB[0])

        # Draw the center of mass and bounding box on the image
        cv2.circle(image,self.cog,10,(0,0,255),2)
        cv2.circle(depth,self.cog,10,(65535),2)
        cv2.rectangle(image,(self.BB[2],self.BB[0]),(self.BB[3],self.BB[1]),(0,0,255),1)
        cv2.rectangle(depth,(self.BB[2],self.BB[0]),(self.BB[3],self.BB[1]),(65535),1)

        # Compute ratio between the original and current depths
        factor = float(self.initDepth) / float(self.depthVal)
        # Compute new size of the bounding rect using the ratio
        BBSize = (round(self.initW*factor),round(self.initH*factor))

        # Compute new bounding box
        self.BB = updateBB(self.cog,BBSize,image.shape)

        # Return images
        return image, depth

    def compute3D(self):
        # Compute 3D coordinates
        # HELP: u = f/z*x; v = f/z*y
        z = self.depthVal / 10
        x = (self.cog[0] - self.imgW/2)*z/self.focal
        y = (self.cog[1] - self.imgH/2)*z/self.focal
        return x,y,z

