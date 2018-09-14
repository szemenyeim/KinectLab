import cv2
from Camera import FolderCam #,Camera
from ImageProcessor import ImageProcessor
import time

if __name__ == '__main__':

    #cam = Camera()
    cam = FolderCam("./vid")

    imageProcessor = ImageProcessor()

    times = 0
    cntr = 0

    while True:

        img,depth = cam.getImages()

        if img is None:
            break

        start = time.time()
        imageProcessor.processImage(img,depth)
        end = time.time()

        times += end-start
        cntr += 1

        if cv2.waitKey(1) == 27:
            break

    avgTime = times / cntr
    print("Average running time: %f" % (avgTime * 1000))
    print("Average fps: %f" % (1.0 / avgTime))