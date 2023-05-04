"""
PICTURES/IMAGES MUST BE RESIZED TO 1920x1080
"""



import cv2 
import matplotlib.pyplot as plt 
import numpy as np 

from util import get_parking_spots_bboxes, empty_or_not 

# paths to images for mask and lot
mask = './images/masks/eng12_mask.jpg'
lot = './images/lot_images/eng12.jpg'
# read in images in grayscale
mask = cv2.imread(mask, 0)
lot = cv2.imread(lot)

connected_components = cv2.connectedComponentsWithStats(mask, 16, cv2.CV_32S)
spots = get_parking_spots_bboxes(connected_components)
#@print(spots[0])

for spot in spots:
    x1, y1, w, h = spot 
    # check to see what's in the area of interest 
    spot_crop = lot[y1:y1 + h, x1:x1 + w, :]
    # is spot empty or not
    spot_stuff = empty_or_not(spot_crop)
    if spot_stuff:
        lot = cv2.rectangle(lot, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 2)
    else:
        lot = cv2.rectangle(lot, (x1, y1), (x1 + w, y1 + h), (0, 0, 255), 2)

    # blue rectangle
    #lot = cv2.rectangle(lot, (x1, y1), (x1 + w, y1 + h), (255, 0, 0), 2)

cv2.imshow('lot', lot)
cv2.imshow('mask', mask)
cv2.waitKey(0)