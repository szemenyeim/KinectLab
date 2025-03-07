import numpy as np
import cv2
import os.path as osp
import glob
import pyrealsense2 as rs
import re

class FolderCam(object):

    def __init__(self, path):
        self.path = path

        self.images = sorted(glob.glob1(path, "*.png"))

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, i):
        img_p = osp.join(self.path, self.images[i])
        return cv2.imread(img_p)

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
            return None, None

        # Convert images to numpy arrays
        depth = np.asanyarray(depth_frame.get_data())
        img = np.asanyarray(color_frame.get_data())

        return img, depth