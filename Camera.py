import numpy as np
import cv2
import pyrealsense2 as rs
import os.path as osp
from glob import glob1
import re


def tryint(s):
    try:
        return int(s)
    except:
        return s

def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [ tryint(c) for c in re.split('([0-9]+)', s) ]

class Camera(object):
    def __init__(self):
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        align_to = rs.stream.color
        self.align = rs.align(align_to)

        # Start streaming
        self.pipeline.start(config)

    def getImages(self):
        frames = self.pipeline.wait_for_frames()
        frames = self.align.process(frames)
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            return

        # Convert images to numpy arrays
        depth = np.asanyarray(depth_frame.get_data())
        img = np.asanyarray(color_frame.get_data())

        return img, depth

class FolderCam(object):
    def __init__(self, path):
        self.rgbPath = osp.join(path,"rgb")
        self.depthPath = osp.join(path,"depth")

        self.images = sorted(glob1(self.rgbPath,"*.jpg"), key=alphanum_key)
        if len(self.images) == 0:
            self.images = sorted(glob1(self.rgbPath, "*.png"), key=alphanum_key)
        if len(self.images) == 0:
            raise ValueError(f"No RGB image can be found at the specified directory={path}")

        self.depths = sorted(glob1(self.depthPath,"*.png"), key=alphanum_key)

        if len(self.depths) == 0:
            raise ValueError(f"No depth image can be found at the specified directory={path}")



        self.cntr = 0

    def getImages(self):

        if self.cntr >= len(self.images):
            return None, None

        imgName = osp.join(self.rgbPath,self.images[self.cntr])
        depthName = osp.join(self.depthPath,self.depths[self.cntr])

        img = cv2.imread(imgName, cv2.IMREAD_UNCHANGED)
        depth = cv2.imread(depthName, cv2.IMREAD_UNCHANGED)

        self.cntr += 1

        return img, depth