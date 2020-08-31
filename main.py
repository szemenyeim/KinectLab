import time
from os.path import abspath, dirname, join

import cv2
import numpy as np

from Camera import FolderCam
from ImageProcessor import ImageProcessor
from KalmanFilter import KalmanFilter

if __name__ == '__main__':

    kf = KalmanFilter()

    main_dir = dirname(abspath(__file__))
    vid_path = join(main_dir, "vid")

    # cam = Camera()
    cam = FolderCam(vid_path)

    imageProcessor = ImageProcessor()

    times = 0
    cntr = 0

    while True:

        """Image processing"""
        img, depth = cam.getImages()

        if img is None:
            break

        start = time.time()
        img, depth = imageProcessor.processImage(img, depth)
        x, y, z = imageProcessor.compute3D()

        end = time.time()

        times += end - start
        cntr += 1

        """Kalman Filter"""
        state, cov = kf.filter([x, y, z])

        """Display"""
        depth = (cv2.cvtColor(depth, cv2.COLOR_GRAY2BGR) // 64).astype('uint8')
        dispImage = np.append(img, depth, axis=1)
        dispImage = np.append(dispImage, (np.ones([250, dispImage.shape[1], 3]) * 255).astype('uint8'), axis=0)

        cv2.putText(dispImage, (f"X position = {x:.4f}/KF = {state[0]:.4f}"), (20, 530), cv2.FONT_HERSHEY_COMPLEX, 1,
                    (0, 0, 255))
        cv2.putText(dispImage, (f"Y position = {y:.4f}/KF = {state[1]:.4f}"), (20, 580), cv2.FONT_HERSHEY_COMPLEX, 1,
                    (0, 0, 255))
        cv2.putText(dispImage, (f"Z position = {z:.4f}/KF = {state[2]:.4f}"), (20, 630), cv2.FONT_HERSHEY_COMPLEX, 1,
                    (0, 0, 255))

        cv2.imshow("Tracking", dispImage)

        if cv2.waitKey(1) == 27:
            break

    avgTime = times / cntr
    print("Average running time: %f" % (avgTime * 1000))
    print("Average fps: %f" % (1.0 / avgTime))
