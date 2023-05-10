import cv2
import pickle
import cvzone
import numpy as np

cap = cv2.VideoCapture('video.mp4')

with open('CarParkPos', 'rb') as f:
    posList = pickle.load(f)

width, height = 60, 39

def checkParkingSpace(imgProcessed):
    for pos in posList:
        x,y = pos

        imgCrop = imgProcessed[y:y + height, x:x + width]
        cv2.imshow(str(x * y), imgCrop)

        count = cv2.countNonZero(imgCrop)
        cvzone.putTextRect(img, str(count), (x,y+height-10), scale=1, thickness=1, offset=-1)

        if count < 600:
            color = (0,255,0)
        else:
            color = (0,0,255)

        cv2.rectangle(img, pos, (pos[0] + width, pos[1] + height), color, 2)

while True:

    if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    success, img = cap.read()

    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (3,3), 1)
    imgThreshold = cv2.adaptiveThreshold(imgBlur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY_INV, 25, 16)
    imgMedian = cv2.medianBlur(imgThreshold, 5) #Maybe dont use this
    kernel = np.ones((3,3), np.uint8)
    imgDilate = cv2.dilate(imgMedian, kernel, iterations=1)

    checkParkingSpace(imgDilate)

    cv2.imshow("Image", img)
    cv2.waitKey(10)