import cv2
from Camera import FolderCam ,Camera
from ImageProcessor import ImageProcessor
import numpy as np
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
        img,depth = imageProcessor.processImage(img,depth)
        x,y,z = imageProcessor.compute3D()
        end = time.time()

        times += end-start
        cntr += 1

        depth = (cv2.cvtColor(depth,cv2.COLOR_GRAY2BGR)//64).astype('uint8')
        dispImage = np.append(img,depth,axis = 1)
        dispImage = np.append(dispImage,(np.ones([250,1280,3])*255).astype('uint8'), axis=0)

        cv2.putText(dispImage,("X position = %f" % x), (20, 530), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255))
        cv2.putText(dispImage,("Y position = %f" % y), (20, 580), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255))
        cv2.putText(dispImage,("Z position = %f" % z), (20, 630), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255))

        cv2.imshow("Tracking",dispImage)

        if cv2.waitKey(1) == 27:
            break

    avgTime = times / cntr
    print("Average running time: %f" % (avgTime * 1000))
    print("Average fps: %f" % (1.0 / avgTime))