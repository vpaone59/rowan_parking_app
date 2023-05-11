import cv2
import cvzone
import numpy as np

def checkParkingSpace(img, width, height, posList):
        imgProcessed = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        imgProcessed = cv2.GaussianBlur(imgProcessed, (3,3), 1)
        imgProcessed = cv2.adaptiveThreshold(imgProcessed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                             cv2.THRESH_BINARY_INV, 25, 16)
        imgProcessed = cv2.medianBlur(imgProcessed, 5) #Maybe dont use this
        kernel = np.ones((3,3), np.uint8)
        imgProcessed = cv2.dilate(imgProcessed, kernel, iterations=1)

        for pos in posList:
            x,y = pos

            imgCrop = imgProcessed[y:y + height, x:x + width]
            #cv2.imshow(str(x * y), imgCrop)

            count = cv2.countNonZero(imgCrop)
            cvzone.putTextRect(img, str(count), (x,y+height-10), scale=1, thickness=1, offset=-1)

            if count < 600:
                color = (0,255,0)
            else:
                color = (0,0,255)

            cv2.rectangle(img, pos, (pos[0] + width, pos[1] + height), color, 2)