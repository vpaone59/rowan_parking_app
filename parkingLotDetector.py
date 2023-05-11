import cv2
import pickle
import numpy as np
from util import checkParkingSpace

class ParkingLotDetector:
    def __init__(self, video_path, pos_filename):
        self.cap = cv2.VideoCapture(video_path)

        with open(pos_filename, 'rb') as f:
            self.posList = pickle.load(f)

        self.width, self.height = 60, 39


    def run(self):
        while True:
            if self.cap.get(cv2.CAP_PROP_POS_FRAMES) == self.cap.get(cv2.CAP_PROP_FRAME_COUNT):
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            success, img = self.cap.read()

            checkParkingSpace(img, self.width, self.height, self.posList)

            cv2.imshow("Image", img)
            # Check for key press
            key = cv2.waitKey(10)
            if key == ord('q'): # Press 'q' to quit
                break

        # Clean up
        cv2.destroyAllWindows()
        self.cap.release()