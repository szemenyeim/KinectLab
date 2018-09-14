import cv2

class BBGUI(object):
    def __init__(self):
        self.sPoint = None
        self.ePoint = None
        self.btnDown = False
        self.image = None

    def on_mouse(self, event, x, y, flags, params):

        if event == cv2.EVENT_LBUTTONDOWN:
            self.sPoint = (x, y)
            self.btnDown = True

        elif event == cv2.EVENT_MOUSEMOVE and self.btnDown:
            self.ePoint = (x, y)
            img_draw = self.image.copy()
            cv2.rectangle(img_draw, self.sPoint, self.ePoint, (0, 0, 255), 1)
            cv2.imshow("Object Selection", img_draw)

        elif event == cv2.EVENT_LBUTTONUP:
            self.ePoint = (x, y)
            self.btnDown = False
            img_draw = self.image.copy()
            cv2.rectangle(img_draw, self.sPoint, self.ePoint, (0, 0, 255), 1)
            cv2.imshow("Object Selection", img_draw)

    def getBB(self,image):
        self.image = image
        cv2.imshow("Object Selection", image)
        cv2.setMouseCallback("Object Selection", self.on_mouse, 0)
        while cv2.waitKey(1) != 13 or (self.ePoint is None or self.sPoint is None):
            pass
        BB = [self.sPoint[1], self.ePoint[1], self.sPoint[0], self.ePoint[0]]
        cv2.destroyWindow("Object Selection")
        return BB