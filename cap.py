import cv2
import time


# Const
webcam_source_num = 1


# Capture class
class Capture():
    def __init__(self, parent):
        self.parent = parent
        self.cap = cv2.VideoCapture(webcam_source_num)
        self.frame = None

    def update(self):
        while True:
            ret, frame = self.cap.read()
            if ret:
                self.frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)